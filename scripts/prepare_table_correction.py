"""Bounded CPU preparation: only original train pages with complete HTML tables."""
import argparse,concurrent.futures,hashlib,json,os,random,shutil,time
from pathlib import Path
import boto3
from botocore.config import Config
from annotation_targets import resolve_annotation,TARGET_FORMAT,TableTargetError
from inspect_training_tables import annotation_range,range_bytes,classify,BUCKET
from minio_retry import read_object,object_sha256,retryable,RETRY_DELAYS
from table_correction_runtime import SOURCE_DATA_SHA,encode,digest,MAX_SEQUENCE

def write(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--model-dir',required=True)
    parser.add_argument('--source-complete',type=Path,required=True);args=parser.parse_args()
    root=args.root;data=root/'data'
    if root.exists():raise ValueError('Corrective phase path already exists; inspect it, do not overwrite')
    root.mkdir(parents=True);data.mkdir()
    if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('At least 8 GiB free required; no files deleted')
    s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',
                   config=Config(connect_timeout=10,read_timeout=30,retries={'max_attempts':0},s3={'addressing_style':'path'}))
    source_bytes=args.source_complete.read_bytes();assert digest(args.source_complete)==SOURCE_DATA_SHA
    assert read_object(s3,BUCKET,'dataset/manifests/ocr-manifest-v1/COMPLETE.json')==source_bytes
    source=json.loads(source_bytes);entries=[e for e in source['shards'] if e['shard'].startswith('data/')][:8]
    assert len(entries)==8
    pools={};source_manifests=[];scan_counts={};rejections=[];accepted={k:[] for k in ['train','validation-tables','validation-control','test-tables']}
    for index,entry in enumerate(entries):
        m=entry['manifest'];raw=read_object(s3,BUCKET,m['key']);assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
        groups={k:[] for k in accepted};counts={'scanned':0,'table_candidates':0}
        for line in raw.splitlines():
            row=json.loads(line);assert not row['augmented'];counts['scanned']+=1
            assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
            flags=classify(row['text']);table=bool(flags['table_links'] or flags['html_tables'] or flags['pipe_heuristic'])
            counts['table_candidates']+=int(table)
            category='train' if row['split']=='train' and table else 'validation-tables' if row['split']=='validation' and table else 'validation-control' if row['split']=='validation' and not table else 'test-tables' if row['split']=='test' and table else None
            if category:groups[category].append(row)
        for category,rows in groups.items():random.Random(42+index).shuffle(rows)
        pools[entry['shard']]=groups;source_manifests.append(m);scan_counts[entry['shard']]=counts
    from transformers import AutoProcessor
    import torch
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    sizes={e['shard']:s3.head_object(Bucket=BUCKET,Key='dataset/raw/'+e['shard'])['ContentLength'] for e in entries}
    def fetch(row,shard):
        for attempt in range(4):
            try:
                ann,offset=annotation_range(s3,row,sizes[shard]);im=row['image']
                image=range_bytes(s3,im['key'],im['offset'],im['size'],sizes[shard])
                assert hashlib.sha256(image).hexdigest()==im['sha256']
                annotation=json.loads(ann);assert annotation['markdown']==row['text']
                return ann,annotation,image
            except Exception as exc:
                if not retryable(exc) or attempt==3:raise
                print('PREP_RETRY',json.dumps({'id':row['id'],'attempt':attempt+1,'error_type':type(exc).__name__}),flush=True);time.sleep(RETRY_DELAYS[attempt])
    targets={'train':512,'validation-tables':8,'validation-control':4,'test-tables':2}
    try:
        for entry in entries:
            shard=entry['shard']
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for category,target in targets.items():
                    count=0;candidates=pools[shard][category]
                    # Fetch groups of four; do not submit the whole corpus at once.
                    for start in range(0,min(len(candidates),target*2+32),4):
                        group=candidates[start:start+4]
                        futures=[pool.submit(fetch,row,shard) for row in group]
                        for row,future in zip(group,futures):
                            ann_bytes,ann,image=future.result()
                            if count>=target:continue
                            try:
                                if ann.get('meta',{}).get('language')!='ar':raise ValueError('non_arabic_page')
                                resolved=resolve_annotation(ann)
                                if category=='validation-control':
                                    if resolved.audit['table_count']!=0:raise ValueError('control_contains_tables')
                                elif resolved.audit['table_count']<1:raise ValueError('no_verified_html_table')
                                corrected={**row,'schema':'table-correction-pilot-v1','text':resolved.text,
                                           'text_sha256':hashlib.sha256(resolved.text.encode()).hexdigest(),'target_format':TARGET_FORMAT,
                                           'table_audit':resolved.audit,'annotation_sha256':hashlib.sha256(ann_bytes).hexdigest(),
                                           'source_markdown_sha256':row['text_sha256'],'language':'ar'}
                                batch=encode(processor,corrected,image)
                                corrected['sequence_tokens']=batch['input_ids'].shape[1]
                                corrected['supervised_tokens']=int((batch['labels'][:,1:]!=-100).sum());del batch
                            except (TableTargetError,ValueError) as exc:
                                rejections.append({'id':row['id'],'category':category,'reason':str(exc)});continue
                            slug=shard.split('/')[-1].removesuffix('.tar')+'_'+row['id'].split(':')[-1]
                            im_path=data/'images'/(slug+Path(row['image']['member']).suffix);im_path.parent.mkdir(exist_ok=True);im_path.write_bytes(image)
                            ann_path=data/'annotations'/(slug+'.json');ann_path.parent.mkdir(exist_ok=True);ann_path.write_bytes(ann_bytes)
                            corrected.update(local_image=str(im_path),local_annotation=str(ann_path));accepted[category].append(corrected);count+=1
                            if len(accepted['train'])%128==0 or category!='train':
                                state={'state':'preparing','counts':{k:len(v) for k,v in accepted.items()},'rejections':len(rejections),'shard':shard}
                                write(root/'preparation-status.json',state)
                                if category=='train':print('TABLE_PREPARATION',json.dumps(state),flush=True)
                        if count>=target:break
                    if count!=target:raise ValueError(f'Not enough verified {category} pages in {shard}: {count}/{target}')
            if shutil.disk_usage(root).free<6*1024**3:raise RuntimeError('Preparation disk reserve reached; no files deleted')
        split_groups={k:{r['group_id'] for r in rows} for k,rows in accepted.items()}
        assert not split_groups['train'] & (split_groups['validation-tables']|split_groups['validation-control']|split_groups['test-tables'])
        assert not split_groups['validation-tables'] & split_groups['test-tables']
        manifests={}
        for category,rows in accepted.items():
            path=data/(category+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            manifests[category]={'path':str(path),'sha256':digest(path),'samples':len(rows)}
        (data/'rejections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejections),encoding='utf-8')
        prefix='training-tables-correction-v1/'
        def publish(path):
            key=prefix+path.relative_to(root).as_posix();sha=digest(path)
            for attempt in range(4):
                try:
                    s3.upload_file(str(path),BUCKET,key,ExtraArgs={'Metadata':{'sha256':sha}})
                    assert object_sha256(s3,BUCKET,key)==sha
                    return
                except Exception as exc:
                    if not retryable(exc) or attempt==3:raise
                    time.sleep(RETRY_DELAYS[attempt])
        files=[p for p in data.rglob('*') if p.is_file()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(publish,p) for p in files]
            for n,future in enumerate(futures,1):
                future.result()
                if n%512==0:print('TABLE_DATA_BACKUP',n,len(files),flush=True)
        complete={'schema':'table-correction-pilot-v1','source_data_complete_sha256':SOURCE_DATA_SHA,
                  'source_manifests':source_manifests,'scan_counts':scan_counts,'seed':42,'original_shards':8,
                  'selection':'512 original Arabic train table pages per shard; no augmentations in this initial corrective pass.',
                  'manifests':manifests,'counts':{k:len(v) for k,v in accepted.items()},'rejected_candidates':len(rejections),
                  'target_format':TARGET_FORMAT,'maximum_sequence':MAX_SEQUENCE,
                  'encoder_sha256':digest(Path(__file__).with_name('table_correction_runtime.py')),
                  'target_builder_sha256':digest(Path(__file__).with_name('annotation_targets.py')),
                  'maximum_observed_tokens':max(r['sequence_tokens'] for rows in accepted.values() for r in rows),
                  'training_tables':sum(r['table_audit']['table_count'] for r in accepted['train']),
                  'training_cells':sum(r['table_audit']['cell_count'] for r in accepted['train']),
                  'all_data_files_s3_sha256_verified':True,'s3_prefix':prefix,'sources_modified':False}
        write(data/'COMPLETE.json',complete);publish(data/'COMPLETE.json')
        write(root/'preparation-status.json',{'state':'complete','counts':complete['counts'],'manifest_sha256':digest(data/'COMPLETE.json')})
        print('TABLE_DATA_COMPLETE',json.dumps(complete),flush=True)
    except Exception as exc:
        write(root/'preparation-failure.json',{'state':'failed','error_type':type(exc).__name__,'error':str(exc),'counts':{k:len(v) for k,v in accepted.items()}})
        raise

if __name__=='__main__':main()
