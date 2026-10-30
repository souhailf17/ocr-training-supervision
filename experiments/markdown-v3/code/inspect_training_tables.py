"""CPU-only, read-only audit of table labels in the exact training manifests.

Scan a bounded number of original manifests; recover selected original JSON/PNG
members with byte ranges. Never rewrite a source object or a training manifest.
"""
import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter
from pathlib import Path
from annotation_targets import resolve_annotation, table_references, TableTargetError

BUCKET='qwen35-9b-ocr-finetuning'
KNOWN=['sample_0009924','sample_0009947','sample_0009976','sample_0009984','sample_0009684']

def digest(data):return hashlib.sha256(data).hexdigest()

def classify(text):
    links=len(list(table_references(text)))
    lines=sum(line.count('|')>=2 for line in text.splitlines())
    return {'table_links':links,'html_tables':len(re.findall(r'<table(?:\s|>)',text,re.I)),
            'pipes':text.count('|'),'pipe_heuristic':text.count('|')>=9 and lines>=3}

def range_bytes(s3,key,start,length,total):
    if start<0 or length<=0 or start+length>total:raise ValueError('Unsafe TAR range')
    response=s3.get_object(Bucket=BUCKET,Key=key,Range=f'bytes={start}-{start+length-1}')
    try:data=response['Body'].read(length+1)
    finally:response['Body'].close()
    if len(data)!=length or response.get('ContentRange')!=f'bytes {start}-{start+length-1}/{total}':
        raise ValueError('S3 did not return the exact requested range')
    return data

def header_matches(data,name):
    try:info=tarfile.TarInfo.frombuf(data,'utf-8','surrogateescape')
    except (tarfile.HeaderError,ValueError):return None
    return info if info.isfile() and info.name==name and 0<info.size<=2*1024**2 else None

def annotation_range(s3,row,total):
    image=row['image'];key=image['key'];name=row['annotation_member']
    after=image['offset']+((image['size']+511)//512)*512
    if after+512<=total:
        info=header_matches(range_bytes(s3,key,after,512,total),name)
        if info:return range_bytes(s3,key,after+512,info.size,total),after
    # A JSON immediately before the PNG may have variable size. Read bounded
    # aligned windows, and require a valid TAR checksum and the exact member ID.
    image_header=image['offset']-512
    for budget in (256*1024,2*1024**2+512):
        start=max(0,image_header-budget);start-=start%512
        window=range_bytes(s3,key,start,image_header-start,total)
        for offset in range(0,len(window)-511,512):
            info=header_matches(window[offset:offset+512],name)
            if info:
                header=start+offset
                if header+512+((info.size+511)//512)*512!=image_header:
                    continue
                return range_bytes(s3,key,header+512,info.size,total),header
    raise ValueError('JSON is not adjacent to the manifest-indexed image; refusing to guess: '+row['id'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--complete-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--max-shards',type=int,default=1)
    parser.add_argument('--max-examples',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.max_shards<=3 or not 1<=args.max_examples<=12:raise ValueError('Inspection exceeds bounded limits')
    if args.output_dir.exists():raise ValueError('Refusing to overwrite an inspection')
    import boto3
    from botocore.config import Config
    from PIL import Image
    s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',
       config=Config(connect_timeout=10,read_timeout=30,retries={'max_attempts':3},s3={'addressing_style':'path'}))
    complete_bytes=args.complete_file.read_bytes();complete=json.loads(complete_bytes)
    if complete['schema']!='ocr-manifest-v1':raise ValueError('Expected the original training manifest version')
    response=s3.get_object(Bucket=BUCKET,Key='dataset/manifests/ocr-manifest-v1/COMPLETE.json')
    try:s3_complete=response['Body'].read()
    finally:response['Body'].close()
    if s3_complete!=complete_bytes:raise ValueError('PVC and S3 COMPLETE differ; identify the actual training version before continuing')
    origin={'schema':complete['schema'],'complete_sha256':digest(complete_bytes),'pvc_s3_complete_bytes_match':True,
            'complete_counts':complete['counts'],'s3_bucket':BUCKET,'checkpoint_verified':False}
    if args.checkpoint:
        with args.checkpoint.open('rb') as f:checkpoint_sha=hashlib.file_digest(f,'sha256').hexdigest()
        if not args.checkpoint_sha256 or checkpoint_sha!=args.checkpoint_sha256:raise ValueError('Checkpoint SHA differs')
        import torch
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        recorded=state.get('data_complete_sha256')
        if recorded!=origin['complete_sha256']:raise ValueError('Checkpoint did not use this exact COMPLETE file')
        origin.update(checkpoint_verified=True,checkpoint_sha256=checkpoint_sha,
                      checkpoint_data_complete_sha256=recorded,checkpoint_global_step=state['clock']['step_index'])
        del state
    entries=[e for e in complete['shards'] if e['shard'].startswith('data/')][:args.max_shards]
    rows=[];scans=[];counts=Counter();by_split={};known={};candidates={}
    for entry in entries:
        manifest=entry['manifest']
        if manifest['bytes']>64*1024**2:raise ValueError('Manifest exceeds bounded memory budget')
        response=s3.get_object(Bucket=BUCKET,Key=manifest['key'])
        try:data=response['Body'].read(manifest['bytes']+1)
        finally:response['Body'].close()
        if len(data)!=manifest['bytes'] or digest(data)!=manifest['sha256']:raise ValueError('Manifest SHA mismatch')
        shard_counts=Counter()
        for line in data.splitlines():
            row=json.loads(line);flags=classify(row['text']);split=row['split']
            if row['augmented']:raise ValueError('Original-shard inspection unexpectedly includes augmentation')
            if digest(row['text'].encode())!=row['text_sha256']:raise ValueError('Target SHA mismatch')
            counts['samples']+=1;shard_counts['samples']+=1
            split_counts=by_split.setdefault(split,Counter());split_counts['samples']+=1
            for field in ('table_links','html_tables','pipe_heuristic'):
                if flags[field]:counts[field+'_samples']+=1;shard_counts[field+'_samples']+=1;split_counts[field+'_samples']+=1
            stem=row['id'].rsplit(':',1)[-1]
            row['_manifest_key']=manifest['key'];row['_manifest_sha256']=manifest['sha256'];row['_flags']=flags
            if stem in KNOWN:known[stem]=row
            if flags['table_links']:
                candidates.setdefault(split,[])
                if len(candidates[split])<3:candidates[split].append(row)
            if flags['pipe_heuristic'] and len(candidates.get('pipe_heuristic',[]))<2:
                candidates.setdefault('pipe_heuristic',[]).append(row)
        scans.append({'shard':entry['shard'],'manifest':manifest,'counts':dict(shard_counts)})
    # Guarantee train examples first; known supplied IDs and one validation page
    # remain diagnostic references. Nothing here defines a correction train set.
    queue=[(r,'training-table') for r in candidates.get('train',[])]+[(known[k],'supplied-id-lookup') for k in KNOWN if k in known]
    queue += [(r,'validation-table') for r in candidates.get('validation',[])[:1]]
    queue += [(r,'pipe-heuristic') for r in candidates.get('pipe_heuristic',[])]
    seen=set();selected=[]
    for row,reason in queue:
        if row['id'] not in seen:
            seen.add(row['id']);selected.append((row,reason))
            if len(selected)==args.max_examples:break
    args.output_dir.mkdir(parents=True)
    total_range_bytes=0
    for row,reason in selected:
        image=row['image'];head=s3.head_object(Bucket=BUCKET,Key=image['key']);total=head['ContentLength']
        raw,header=annotation_range(s3,row,total);annotation=json.loads(raw)
        png=range_bytes(s3,image['key'],image['offset'],image['size'],total)
        if digest(png)!=image['sha256']:raise ValueError('Original image SHA differs from training index')
        with Image.open(io.BytesIO(png)) as im:
            im.verify()
        with Image.open(io.BytesIO(png)) as im:
            if im.size!=(image['width'],image['height']):raise ValueError('Original image dimensions differ')
        if annotation.get('markdown')!=row['text']:raise ValueError('Raw JSON Markdown differs from the actual training target')
        stem=row['id'].rsplit(':',1)[-1]
        if not re.fullmatch(r'sample_\d+',stem):raise ValueError('Unsafe source sample ID')
        annotation_path=args.output_dir/'annotations'/(stem+'.json');annotation_path.parent.mkdir(exist_ok=True)
        annotation_path.write_bytes(raw)
        extension=Path(image['member']).suffix.lower()
        image_path=args.output_dir/'images'/(stem+extension);image_path.parent.mkdir(exist_ok=True);image_path.write_bytes(png)
        old=args.output_dir/'old-targets'/(stem+'.md');old.parent.mkdir(exist_ok=True);old.write_text(row['text'],encoding='utf-8')
        record={'id':row['id'],'group_id':row['group_id'],'split':row['split'],'selection_reason':reason,
                'manifest_key':row['_manifest_key'],'manifest_sha256':row['_manifest_sha256'],
                'source_flags':row['_flags'],'raw_key':image['key'],'image':image,
                'annotation_member':row['annotation_member'],'annotation_header_offset':header,
                'annotation_sha256':digest(raw),'annotation_bytes':len(raw),'source_table_entries':len(annotation.get('tables',[])),
                'old_target_sha256':row['text_sha256'],'old_target_equals_raw_markdown':True,
                'image_sha_matches_training_manifest':True,'raw_tar_full_sha_recomputed':False,
                'annotation_file':str(annotation_path.relative_to(args.output_dir)),
                'image_file':str(image_path.relative_to(args.output_dir)),
                'old_target_file':str(old.relative_to(args.output_dir))}
        try:
            resolved=resolve_annotation(annotation)
            new=args.output_dir/'corrected-targets'/(stem+'.md');new.parent.mkdir(exist_ok=True);new.write_text(resolved.text,encoding='utf-8')
            record.update(corrected_target_file=str(new.relative_to(args.output_dir)),corrected_audit=resolved.audit,
                          old_target_chars=len(row['text']),corrected_target_chars=len(resolved.text))
        except TableTargetError as exc:record['target_error']=str(exc)
        rows.append(record);total_range_bytes+=len(raw)+len(png)
        print('TABLE_DATA_SAMPLE',json.dumps({'id':row['id'],'split':row['split'],'tables':record['source_table_entries'],'cells':record.get('corrected_audit',{}).get('cell_count'),'old_target_has_table_links':bool(row['_flags']['table_links'])}),flush=True)
    report={'state':'complete','source':origin,'scanned_shards':scans,'scan_counts':dict(counts),
            'scan_counts_by_split':{s:dict(c) for s,c in by_split.items()},'selected':rows,
            'supplied_ids_found':{k:{'id':r['id'],'split':r['split'],'flags':r['_flags']} for k,r in known.items()},
            'selected_annotation_and_image_bytes':total_range_bytes,
            'limits':'Bounded original-shard inspection, not a full dataset census. Range-read annotations/image hashes checked; full raw TAR digests not recomputed.',
            'gpu_count':0,'training_executed':False,'sources_modified':False}
    (args.output_dir/'inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Small private result bundle, returned through this owned CPU job's logs.
    import base64,zipfile
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in args.output_dir.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(args.output_dir).as_posix())
    archive=buffer.getvalue()
    if len(archive)>20*1024**2:raise ValueError('Result export exceeds its bounded budget')
    print('TABLE_DATA_EXPORT '+json.dumps({'sha256':digest(archive),'bytes':len(archive),'base64':base64.b64encode(archive).decode()}),flush=True)

if __name__=='__main__':main()
