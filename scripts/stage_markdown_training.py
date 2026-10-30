"""CPU final gate and fixed held-out monitoring for the complete v3 corpus."""
import argparse,hashlib,json,heapq,shutil
from pathlib import Path
from collections import Counter
from markdown_table_targets import sha,TARGET_FORMAT,VERSION
from markdown_data_runtime import SOURCE_SHA,PREFIX,BUCKET,s3_client,read,atomic_json
from minio_retry import object_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--configuration',type=Path,required=True);a=p.parse_args();s3=s3_client();config=json.loads(a.configuration.read_text())
    raw=read(s3,PREFIX+'COMPLETE.json');complete=json.loads(raw)
    if complete['schema']!=VERSION or complete['target_format']!=TARGET_FORMAT or not complete['training_ready'] or complete['source_complete_sha256']!=SOURCE_SHA:
        raise ValueError('Complete corrected corpus is not trainable')
    if complete['configuration_sha256']!=sha(a.configuration.read_bytes()):raise ValueError('Corpus configuration changed')
    if sha((a.corpus_root/'COMPLETE.json').read_bytes())!=sha(raw):raise ValueError('Corpus PVC/S3 marker differs')
    if len(complete['shards'])!=206 or complete['counts']['train']!=1841540 or complete['counts']['samples']!=2038290:
        raise ValueError('The selected full-corpus coverage changed')
    pools={k:[] for k in ['validation-tables','validation-control','test-tables','test-control']};caps={'validation-tables':128,'validation-control':128,'test-tables':16,'test-control':16}
    counts=Counter();verified=[]
    for report in complete['shards']:
        m=report['manifest'];data=read(s3,m['key'])
        if sha(data)!=m['sha256'] or len(data)!=m['bytes']:raise ValueError('Completed new manifest changed')
        counts.update(report['counts']);verified.append(m)
        if not report['shard'].startswith('data/'):continue
        for line in data.splitlines():
            row=json.loads(line)
            if row['split'] not in ['validation','test']:continue
            category=row['split']+'-'+('tables' if row['table_count'] else 'control')
            priority=int(sha('42:'+row['group_id']),16);item=(-priority,row['id'],row)
            pool=pools[category]
            if len(pool)<caps[category]:heapq.heappush(pool,item)
            elif item>pool[0]:heapq.heapreplace(pool,item)
    if dict(counts)!=complete['counts']:raise ValueError('Corpus totals differ')
    root=a.run_root
    if (root/'started.json').exists():raise ValueError('Training already started; do not restage it')
    root.mkdir(parents=True,exist_ok=True);(root/'data').mkdir(exist_ok=True)
    if shutil.disk_usage(root).free<12*1024**3:raise ValueError('Need 12 GiB reserve; no existing files deleted')
    parent=Path(config['parent_checkpoint'])
    with parent.open('rb') as f:parent_sha=hashlib.file_digest(f,'sha256').hexdigest()
    if parent_sha!=config['parent_sha256'] or object_sha256(s3,BUCKET,config['parent_s3_key'])!=parent_sha:raise ValueError('Parent checkpoint changed')
    manifests={};seen=set()
    for category,pool in pools.items():
        rows=[item[2] for item in sorted(pool,reverse=True)]
        if len(rows)!=caps[category] or seen & {r['group_id'] for r in rows}:raise ValueError('Held-out monitoring incomplete/overlapping')
        seen.update(r['group_id'] for r in rows)
        body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();path=root/'data'/(category+'.jsonl')
        if path.exists() and path.read_bytes()!=body:raise ValueError('Refusing to replace changed monitoring set')
        path.write_bytes(body);manifests[category]={'path':str(path),'sha256':sha(body),'samples':len(rows)}
        key='training-markdown-v3-orig-aug1/data/'+path.name
        s3.put_object(Bucket=BUCKET,Key=key,Body=body,Metadata={'sha256':sha(body)})
        if read(s3,key)!=body:raise ValueError('Monitoring backup differs')
    marker={'passed':True,'corpus_complete_sha256':sha(raw),'source_complete_sha256':SOURCE_SHA,
            'parent_checkpoint':config['parent_checkpoint'],'parent_sha256':parent_sha,'parent_s3_key':config['parent_s3_key'],
            'train_examples':complete['counts']['train'],'selected_pages':complete['counts']['samples'],
            'all_206_manifest_sha256_verified':True,'monitoring':manifests,'context_limit':8192,
            'optimizer_and_cursor':'fresh for the changed targets and new complete pass; parent checkpoint is preserved'}
    atomic_json(root/'data'/'STAGED.json',marker);atomic_json(root/'data'/'COMPLETE.json',complete)
    print('MARKDOWN_TRAINING_STAGED',json.dumps(marker),flush=True)

if __name__=='__main__':main()
