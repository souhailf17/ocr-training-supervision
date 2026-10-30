"""CPU-only, resumable shard conversion of ALL accepted originals plus aug1.

Real .md objects are readback-verified individually. Images stay in raw TARs;
audits and training manifests are sharded. COMPLETE is never emitted by a worker.
"""
import argparse,concurrent.futures,hashlib,io,json,os,re,shutil,tarfile,time
from collections import Counter,deque
from pathlib import Path,PurePosixPath
from markdown_table_targets import resolve_markdown,TARGET_FORMAT,VERSION,sha
from markdown_data_runtime import (BUCKET,SOURCE_SHA,PREFIX,MAX_SEQUENCE,s3_client,read,publish,
                                  retry,encode,check_table_roundtrip,selected_sources,atomic_json)

def safe_stem(member):
    path=PurePosixPath(member)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source member')
    if not re.fullmatch(r'sample_\d+(?:_01)?',path.stem):raise ValueError('Unexpected sample ID')
    return path.stem

def source_rows(s3,entry):
    m=entry['manifest'];raw=read(s3,m['key'])
    if sha(raw)!=m['sha256'] or len(raw)!=m['bytes']:raise ValueError('Source manifest changed')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=entry['counts']['samples']:raise ValueError('Source count differs')
    result={r['annotation_member']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate source annotation ID')
    if any(sha(r['text'])!=r['text_sha256'] for r in rows):raise ValueError('Source text hash changed')
    return result

def convert_pair(row,annotation_raw,image,processor,raw_json_offset):
    annotation=json.loads(annotation_raw)
    if annotation['markdown']!=row['text']:raise ValueError('Raw JSON differs from the actual historical target')
    if sha(image)!=row['image']['sha256']:raise ValueError('Raw image digest differs')
    resolved=resolve_markdown(annotation);check_table_roundtrip(resolved)
    stem=safe_stem(row['annotation_member']);shard=re.search(r'shard_(\d+)',row['id']).group(1)
    key=PREFIX+('aug1' if row['augmented'] else 'originals')+'/shard_'+shard+'/'+stem+'.md'
    new={**row,'schema':VERSION,'target_format':TARGET_FORMAT,'text':resolved.text,'text_sha256':sha(resolved.text),
         'markdown_object':{'key':key,'sha256':sha(resolved.text),'bytes':len(resolved.text.encode('utf-8'))},
         'source_markdown_sha256':row['text_sha256'],'annotation_sha256':sha(annotation_raw),
         'annotation_offset':raw_json_offset,'annotation_size':len(annotation_raw),
         'table_count':resolved.audit['table_count'],'source_cell_count':resolved.audit['cell_count'],
         'language':annotation.get('meta',{}).get('language')}
    batch=encode(processor,new,image,allow_oversize=True)
    new['sequence_tokens']=int(batch['input_ids'].shape[1]);new['supervised_tokens']=int((batch['labels'][:,1:]!=-100).sum())
    new['training_compatible']=new['sequence_tokens']<=MAX_SEQUENCE
    detail={'id':new['id'],'group_id':row['group_id'],'split':row['split'],
            'annotation_source':{'key':row['image']['key'],'member':row['annotation_member'],
                                 'offset':raw_json_offset,'bytes':len(annotation_raw),'sha256':new['annotation_sha256']},
            'markdown_object':new['markdown_object'],'sequence_tokens':new['sequence_tokens'],
            'supervised_tokens':new['supervised_tokens'],'training_compatible':new['training_compatible'],
            **resolved.audit}
    return new,detail

class DigestReader:
    def __init__(self,body):self.body=body;self.digest=hashlib.sha256();self.size=0
    def read(self,n=-1):
        raw=self.body.read(n);self.digest.update(raw);self.size+=len(raw);return raw

def shard_keys(shard):
    name=shard.replace('/','__').removesuffix('.tar')
    return {kind:PREFIX+kind+'/'+name+suffix for kind,suffix in
            [('manifests','.jsonl'),('audits','.jsonl'),('reports','.json')]}

def maybe_report(s3,key):
    from botocore.exceptions import ClientError
    try:return json.loads(read(s3,key))
    except ClientError as exc:
        if exc.response['Error']['Code'] in {'NoSuchKey','404'}:return None
        raise

def convert_shard(s3,entry,raw_entry,work,processor,configuration_sha,original=None):
    shard=entry['shard'];keys=shard_keys(shard);saved=maybe_report(s3,keys['reports'])
    if saved:
        if saved['configuration_sha256']!=configuration_sha:raise ValueError('Resume code/config changed')
        for name in ['manifest','audit']:
            m=saved[name];raw=read(s3,m['key'])
            if sha(raw)!=m['sha256'] or len(raw)!=m['bytes']:raise ValueError('Previously completed shard changed')
        return saved
    indexed=source_rows(s3,entry);expected=set(indexed);counts=Counter();maximum=0
    if original is not None:
        original_map={r['group_id']:r for r in [json.loads(line) for line in read(s3,original['manifest']['key']).splitlines()]}
        if set(original_map)!={r['group_id'] for r in indexed.values()}:raise ValueError('Original/aug1 group coverage differs')
    else:original_map=None
    manifest=work/'current-manifest.jsonl';audit=work/'current-audit.jsonl';anomalies=work/'anomalies.jsonl'
    pending={};pending_bytes=0;seen=set();found=set();response=s3.get_object(Bucket=BUCKET,Key='dataset/raw/'+shard)
    reader=DigestReader(response['Body']);futures=deque();started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        try:
            with manifest.open('w',encoding='utf-8') as output,audit.open('w',encoding='utf-8') as audits,anomalies.open('w',encoding='utf-8') as errors,tarfile.open(fileobj=reader,mode='r|',bufsize=1024*1024) as archive:
                for member in archive:
                    if member.isdir():continue
                    if not member.isfile():raise ValueError('Unsupported TAR member')
                    stem=safe_stem(member.name);suffix=PurePosixPath(member.name).suffix.lower()
                    role='json' if suffix=='.json' else 'image' if suffix in {'.png','.jpg','.jpeg'} else None
                    if role is None or member.size>32*1024**2:raise ValueError('Unexpected/oversized source member')
                    if stem in seen:raise ValueError('Repeated sample ID in TAR')
                    pair=pending.setdefault(stem,{})
                    if role in pair:raise ValueError('Duplicate source member')
                    handle=archive.extractfile(member);raw=handle.read();handle.close()
                    if len(raw)!=member.size:raise ValueError('Truncated source member')
                    pair[role]={'raw':raw,'name':member.name,'offset':member.offset_data};pending_bytes+=len(raw)
                    if pending_bytes>128*1024**2:raise ValueError('Unpaired TAR buffer exceeds budget')
                    if len(pair)<2:continue
                    counts['samples_seen']+=1;name=pair['json']['name']
                    if name in indexed:
                        row=indexed[name]
                        if pair['image']['name']!=row['image']['member'] or pair['image']['offset']!=row['image']['offset'] or len(pair['image']['raw'])!=row['image']['size']:
                            raise ValueError('Raw image address differs from historical manifest')
                        try:
                            new,detail=convert_pair(row,pair['json']['raw'],pair['image']['raw'],processor,pair['json']['offset'])
                        except Exception as exc:
                            errors.write(json.dumps({'id':row['id'],'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False)+'\n');errors.flush()
                            raise
                        if original_map is not None:
                            parent=original_map[row['group_id']]
                            if (parent['text_sha256'],parent['split'])!=(new['text_sha256'],new['split']):
                                raise ValueError('Aug1 complete target/split differs from original')
                        # The upload queue is bounded; every real .md is reread
                        # and hashed before the manifest report can be committed.
                        futures.append(pool.submit(publish,s3,new['markdown_object']['key'],new['text'].encode('utf-8')))
                        if len(futures)>=32:futures.popleft().result()
                        output.write(json.dumps(new,ensure_ascii=False)+'\n');audits.write(json.dumps(detail,ensure_ascii=False)+'\n')
                        found.add(name);counts['samples']+=1;counts[new['split']]+=1
                        counts['pages_with_tables']+=int(new['table_count']>0);counts['tables']+=new['table_count'];counts['source_cells']+=new['source_cell_count']
                        counts['over_context_limit']+=int(not new['training_compatible']);maximum=max(maximum,new['sequence_tokens'])
                    else:counts['historical_rejections']+=1
                    pending_bytes-=sum(len(v['raw']) for v in pair.values());del pending[stem];seen.add(stem)
                    if counts['samples_seen']%500==0:
                        state={'state':'converting','shard':shard,'counts':dict(counts),'elapsed_seconds':time.time()-started,'maximum_sequence_tokens':maximum}
                        atomic_json(work/'status.json',state);print('MARKDOWN_PREP_METRIC',json.dumps(state),flush=True)
                if pending or found!=expected:raise ValueError('Source accepted-page coverage incomplete')
                for f in futures:f.result()
            while reader.read(8*1024**2):pass
        except Exception:
            if anomalies.exists():publish(s3,PREFIX+'failures/'+shard.replace('/','__')+'.jsonl',anomalies.read_bytes())
            raise
        finally:response['Body'].close()
    if reader.size!=raw_entry['size'] or reader.digest.hexdigest()!=entry['raw_sha256'] or reader.digest.hexdigest()!=raw_entry['lfs']['oid']:
        raise ValueError('Raw HF TAR SHA/size mismatch')
    if counts['samples']!=entry['counts']['samples'] or counts['samples_seen']!=entry['counts']['samples_seen']:
        raise ValueError('New/source page count mismatch')
    for split in ['train','validation','test']:
        if counts[split]!=entry['counts'][split]:raise ValueError('Split membership changed')
    report={'state':'complete','shard':shard,'counts':dict(counts),'configuration_sha256':configuration_sha,
            'source_manifest':entry['manifest'],'raw_sha256':reader.digest.hexdigest(),'raw_bytes':reader.size,
            'manifest':publish(s3,keys['manifests'],manifest.read_bytes()),
            'audit':publish(s3,keys['audits'],audit.read_bytes()),'maximum_sequence_tokens':maximum,
            'all_markdown_objects_readback_verified':True,'elapsed_seconds':time.time()-started}
    publish(s3,keys['reports'],json.dumps(report).encode())
    print('MARKDOWN_SHARD_COMPLETE',json.dumps(report),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['worker','finalize','pilot'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--model-dir',required=True)
    parser.add_argument('--source-complete',type=Path,required=True);parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();s3=s3_client();source_bytes=args.source_complete.read_bytes()
    if sha(source_bytes)!=SOURCE_SHA or read(s3,'dataset/manifests/ocr-manifest-v1/COMPLETE.json')!=source_bytes:
        raise ValueError('Historical source provenance differs')
    complete=json.loads(source_bytes);pairs=selected_sources(complete);tree={e['path']:e for e in json.loads(args.tree.read_text()) if e['type']=='file'}
    config_bytes=args.configuration.read_bytes();config=json.loads(config_bytes);configuration_sha=sha(config_bytes)
    if config['source_sha256']!=SOURCE_SHA or config['target_format']!=TARGET_FORMAT or config['prefix']!=PREFIX:
        raise ValueError('Unexpected new corpus configuration')
    for name in ['markdown_table_targets.py','markdown_data_runtime.py','prepare_markdown_corpus.py','annotation_targets.py']:
        if sha(Path(__file__).with_name(name).read_bytes())!=config['code_sha256'][name]:raise ValueError('Preparation code differs from pinned configuration')
    old=maybe_report(s3,PREFIX+'CONFIG.json')
    if old is not None and old!=config:raise ValueError('Prefix already belongs to different configuration')
    publish(s3,PREFIX+'CONFIG.json',config_bytes)
    if args.action=='finalize':
        reports=[];counts=Counter();maximum=0
        for pair in pairs:
            for entry in pair:
                report=maybe_report(s3,shard_keys(entry['shard'])['reports'])
                if not report or report['configuration_sha256']!=configuration_sha:raise ValueError('Cannot finalize incomplete/different shard')
                if report['source_manifest']!=entry['manifest'] or not report['all_markdown_objects_readback_verified']:raise ValueError('Incomplete provenance/readback')
                counts.update(report['counts']);maximum=max(maximum,report['maximum_sequence_tokens']);reports.append(report)
        result={'schema':VERSION,'target_format':TARGET_FORMAT,'prefix':PREFIX,'state':'complete',
                'source_complete_sha256':SOURCE_SHA,'hf_revision':'4cf05556e21bc54aca54a8b1f81ce172f3ef0477',
                'configuration_sha256':configuration_sha,'counts':dict(counts),'shards':reports,
                'maximum_sequence':MAX_SEQUENCE,'maximum_observed_sequence':maximum,
                'training_ready':counts['over_context_limit']==0,'all_markdown_objects_readback_verified':True,
                'selection':'all historically accepted originals + their aug1 only; no aug2/aug3',
                'split_policy':'inherit v1 original/group split unchanged; validation and test are never trained',
                'source_files_modified':False}
        atomic_json(args.root/'COMPLETE.json',result);publish(s3,PREFIX+'COMPLETE.json',json.dumps(result,ensure_ascii=False,indent=2).encode())
        print('MARKDOWN_CORPUS_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='shards'}),flush=True)
        if not result['training_ready']:raise ValueError('Full Markdown saved, but oversize pages require a policy before training')
        return
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1);processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    if args.action=='pilot':
        from inspect_training_tables import annotation_range,range_bytes
        from minio_retry import object_sha256
        parent=Path(config['parent_checkpoint'])
        if sha(parent.read_bytes())!=config['parent_sha256'] or object_sha256(s3,BUCKET,config['parent_s3_key'])!=config['parent_sha256']:
            raise ValueError('Parent PVC/S3 checkpoint mismatch')
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['clock']['step_index']!=14355 or state['data_complete_sha256']!=SOURCE_SHA:raise ValueError('Parent provenance differs')
        work=args.root/'pilot';work.mkdir(parents=True,exist_ok=True);results=[];originals={}
        ids=['sample_0000003','sample_0000005','sample_0000007','sample_0000267','sample_0009684','sample_0009924','sample_0009947','sample_0009976','sample_0009984']
        for entry in pairs[0]:
            rows=source_rows(s3,entry);total=tree[entry['shard']]['size']
            for row in rows.values():
                stem=safe_stem(row['annotation_member']);base=stem.removesuffix('_01')
                if base not in ids:continue
                ann,header=retry(lambda:annotation_range(s3,row,total));im=row['image']
                image=retry(lambda:range_bytes(s3,im['key'],im['offset'],im['size'],total))
                new,audit=convert_pair(row,ann,image,processor,header+512)
                if not new['training_compatible']:raise ValueError('Pilot target does not fit unchanged context')
                if row['augmented']:
                    if originals[base]!=(new['text_sha256'],new['split']):raise ValueError('Pilot augmented target/group differs')
                else:originals[base]=(new['text_sha256'],new['split'])
                (work/(stem+'.md')).write_text(new['text'],encoding='utf-8');(work/(stem+'.png')).write_bytes(image)
                atomic_json(work/(stem+'.audit.json'),audit);results.append({'id':new['id'],'tables':new['table_count'],'cells':new['source_cell_count'],'tokens':new['sequence_tokens'],'split':new['split']})
        if len(results)!=18:raise ValueError('Pilot selection incomplete')
        result={'passed':True,'parent_pvc_s3_sha_verified':True,'configuration_sha256':configuration_sha,'examples':results,
                'actual_processor_no_truncation':True,'GFM_rendered_cells_roundtrip_exact':True,'original_aug1_targets_splits_match':True}
        atomic_json(work/'result.json',result);publish(s3,PREFIX+'pilot/result.json',json.dumps(result).encode())
        print('MARKDOWN_PILOT_PASSED',json.dumps(result),flush=True);return
    pilot=maybe_report(s3,PREFIX+'pilot/result.json')
    if not pilot or not pilot['passed'] or pilot['configuration_sha256']!=configuration_sha:raise ValueError('Pilot has not passed')
    index=int(os.environ['JOB_COMPLETION_INDEX']);work=args.root/('worker-'+str(index));work.mkdir(parents=True,exist_ok=True)
    if not 0<=index<args.workers:raise ValueError('Worker index invalid')
    if shutil.disk_usage(work).free<6*1024**3:raise ValueError('Insufficient PVC reserve; no files deleted')
    try:
        for n,pair in enumerate(pairs):
            if n%args.workers!=index:continue
            original=None
            for entry in pair:
                report=retry(lambda:convert_shard(s3,entry,tree[entry['shard']],work,processor,configuration_sha,original))
                if original is None:original=report
        atomic_json(work/'result.json',{'state':'complete','worker':index});print('MARKDOWN_WORKER_COMPLETE',index,flush=True)
    except Exception as exc:
        result={'state':'failed','worker':index,'error_type':type(exc).__name__,'error':str(exc)}
        atomic_json(work/'failure.json',result);publish(s3,PREFIX+f'failures/worker-{index}.json',json.dumps(result).encode());raise

if __name__=='__main__':main()
