"""Prepare owned PVC inputs, audit manifests, and inspect installed training APIs."""
import argparse,hashlib, inspect, json, os
from pathlib import Path
import boto3
from botocore.config import Config

ROOT = Path('/work/q35ocr-20260921-p560')
BUCKET = 'qwen35-9b-ocr-finetuning'
PROJECT = 'q35ocr-20260921-p560'
DATA_DIR = Path(os.environ.get('Q35OCR_DATA_DIR', str(ROOT/'data')))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-version',choices=['ocr-manifest-v1','ocr-manifest-v2-tables'],default='ocr-manifest-v1')
    args=parser.parse_args()
    data_dir=DATA_DIR if args.manifest_version=='ocr-manifest-v1' else ROOT/'data-tables-v2'
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['model', 'data', 'logs', 'checkpoints', 'cache']:
        (ROOT/name).mkdir(exist_ok=True)
    data_dir.mkdir(parents=True,exist_ok=True)
    s3 = None
    for endpoint in [os.environ['AWS_ENDPOINT_URL'], 'http://q35ocr-20260921-p560-s3-relay.mtrna-ns1.svc:9000']:
        try:
            candidate = boto3.client('s3', endpoint_url=endpoint, region_name='us-east-1', config=Config(connect_timeout=5, read_timeout=60, retries={'max_attempts':1}, s3={'addressing_style':'path'}))
            owner = json.loads(candidate.get_object(Bucket=BUCKET, Key='documentation/project-owner.json')['Body'].read())
            if PROJECT not in json.dumps(owner):
                raise RuntimeError('Bucket ownership mismatch')
            s3 = candidate
            (ROOT/'endpoint.txt').write_text(endpoint)
            break
        except Exception as exc:
            print('Endpoint check:', endpoint, type(exc).__name__, getattr(exc,'response',{}).get('Error',{}), flush=True)
    if s3 is None:
        raise RuntimeError('Project storage unreachable')
    summary = json.loads(s3.get_object(Bucket=BUCKET, Key=f'dataset/manifests/{args.manifest_version}/COMPLETE.json')['Body'].read())
    if args.manifest_version=='ocr-manifest-v2-tables':
        from annotation_targets import TARGET_FORMAT
        assert summary['target_format']==TARGET_FORMAT
    assert len(summary['shards']) == 412
    checked = 0
    for report in summary['shards']:
        for name in ['manifest', 'rejections']:
            item = report[name]
            head = s3.head_object(Bucket=BUCKET, Key=item['key'])
            assert head['ContentLength'] == item['bytes']
            assert head['Metadata']['sha256'] == item['sha256']
            checked += 1
    (data_dir/'COMPLETE.json').write_text(json.dumps(summary))
    print('Prepared object metadata verified:', checked, flush=True)
    tree = json.loads(Path('/project/model-tree.json').read_text())
    for item in tree:
        if item['type'] != 'file': continue
        path = item['path']
        record = json.loads(s3.get_object(Bucket=BUCKET, Key='dataset/manifests/transfers/model/'+hashlib.sha256(path.encode()).hexdigest()+'.json')['Body'].read())
        assert record['status'] in ['verified','verified-existing']
        target = ROOT/'model'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        def digest(p):
            with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
        if not target.exists() or digest(target) != record['checksum']:
            temp = target.with_suffix(target.suffix+'.partial')
            s3.download_file(BUCKET, 'model/base/'+path, str(temp))
            assert temp.stat().st_size == item['size'] and digest(temp) == record['checksum']
            temp.replace(target)
        print('Model verified on PVC:', path, flush=True)
    rows = []; groups = set(); counts = {'train':0, 'validation':0, 'test':0}
    for report in summary['shards']:
        body = s3.get_object(Bucket=BUCKET, Key=report['manifest']['key'])['Body']
        try:
            for line in body.iter_lines():
                r = json.loads(line)
                if counts[r['split']] >= 8 or r['group_id'] in groups: continue
                im = r['image']; start = im['offset']
                data = s3.get_object(Bucket=BUCKET, Key=im['key'], Range=f"bytes={start}-{start+im['size']-1}")['Body'].read()
                assert hashlib.sha256(data).hexdigest() == im['sha256']
                assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
                dest = data_dir/f'{len(rows):03d}.image'
                dest.write_bytes(data); r['local_image'] = str(dest)
                rows.append(r); groups.add(r['group_id']); counts[r['split']] += 1
                if min(counts.values()) == 8: break
        finally: body.close()
        if min(counts.values()) == 8: break
    assert min(counts.values()) == 8
    (data_dir/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    if args.manifest_version=='ocr-manifest-v2-tables':
        # Same held-out pages as before; new answers, never new split membership.
        from refresh_table_selection import refresh_selection
        selected=ROOT/'pilot-10h-v1/validation.jsonl'
        if selected.exists():
            refresh_selection(s3,BUCKET,summary,selected,data_dir/'validation.jsonl')
    import importlib.metadata as metadata
    from arsenal.config.optimizer import OptimizerConfig
    from arsenal.config.distributed import DistributedConfig
    from arsenal.config.dataloader import BaseDataloaderConfig
    from lr_scheduler_basic import CosineSchedulerConfig
    from arsenal.model import ArsenalModule
    from arsenal.trainer import BaseTrainer
    api_name='training-api.txt' if args.manifest_version=='ocr-manifest-v1' else 'training-api-tables-v2.txt'
    with (ROOT/'logs'/api_name).open('w') as out:
        for cls in [OptimizerConfig, DistributedConfig, BaseDataloaderConfig, CosineSchedulerConfig, ArsenalModule, BaseTrainer]:
            out.write('\n'+inspect.getsource(cls))
        out.write('\nPACKAGES\n'+json.dumps({d.metadata['Name']:d.version for d in metadata.distributions()}))
    result = {'status':'passed', 'model_files':len([i for i in tree if i['type']=='file']), 'prepared_objects_checked':checked, 'smoke_samples':counts,'schema':args.manifest_version,'data_directory':str(data_dir)}
    log_name='staging-result.json' if args.manifest_version=='ocr-manifest-v1' else 'staging-result-tables-v2.json'
    (ROOT/'logs'/log_name).write_text(json.dumps(result))
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
