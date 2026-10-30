"""Stream verified raw TARs into range-addressed OCR manifests, one shard at a time.

Original/augmentation groups and duplicate image/target hashes are tracked in a
transactional SQLite index. Resume is at shard boundaries. Raw files are unchanged.
"""
import argparse,fcntl,hashlib,io,json,os,re,shutil,sqlite3,tarfile,unicodedata
from collections import Counter
from pathlib import Path,PurePosixPath
from project_runtime import ROOT,load_credentials,now,atomic_json,put_verified
from s3_common import client,BUCKET
from PIL import Image
from boto3.s3.transfer import TransferConfig

VERSION='ocr-manifest-v1'
TARGET_BUILDER=None
SEED=42
Image.MAX_IMAGE_PIXELS=40_000_000

def sha(data): return hashlib.sha256(data).hexdigest()

def choose_split(text):
    # Equal normalized labels and all unchanged augmented labels stay together.
    normalized=unicodedata.normalize('NFC',text).replace('\r\n','\n').strip()
    value=int(sha(f'{SEED}\0{normalized}'.encode())[:16],16)%10000
    return 'train' if value<9000 else 'validation' if value<9500 else 'test'

def group_id(shard,stem):
    m=re.fullmatch(r'data/shard_(\d+)\.tar',shard)
    if m:
        if not re.fullmatch(r'sample_\d+',stem): raise ValueError('Unknown original sample naming: '+stem)
        return m[1]+':'+stem,False
    m=re.fullmatch(r'data_aug/shard_(\d+)_aug([123])\.tar',shard)
    if not m: raise ValueError('Unknown shard naming: '+shard)
    suffix=f'_0{m[2]}'
    if not stem.endswith(suffix) or not re.fullmatch(r'sample_\d+',stem[:-len(suffix)]):
        raise ValueError('Unknown augmented sample naming: '+stem)
    return m[1]+':'+stem[:-len(suffix)],True

class DigestReader:
    def __init__(self,body): self.body=body; self.hash=hashlib.sha256(); self.size=0
    def read(self,n=-1):
        data=self.body.read(n); self.hash.update(data); self.size+=len(data); return data

def upload_manifest(s3,path,key):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    s3.upload_file(str(path),BUCKET,key,ExtraArgs={'Metadata':{'sha256':digest.hexdigest(),'schema':VERSION}},
        Config=TransferConfig(multipart_threshold=16*1024*1024,multipart_chunksize=16*1024*1024,use_threads=False))
    response=s3.get_object(Bucket=BUCKET,Key=key)
    check=hashlib.sha256(); size=0
    try:
        for chunk in response['Body'].iter_chunks(8*1024*1024): check.update(chunk); size+=len(chunk)
    finally: response['Body'].close()
    if size!=path.stat().st_size or check.hexdigest()!=digest.hexdigest(): raise ValueError('Manifest upload verification failed')
    return {'key':key,'bytes':size,'sha256':digest.hexdigest()}

def create_db(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY,text_hash TEXT NOT NULL,split TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS images (hash TEXT PRIMARY KEY,text_hash TEXT NOT NULL,split TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS rejected_documents (id TEXT PRIMARY KEY,reason TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY,report TEXT NOT NULL);
    ''')
    revision=json.loads((ROOT/'evidence/source-summary.json').read_text())['dataset']['revision']
    config=json.dumps({'version':VERSION,'seed':SEED,'revision':revision},sort_keys=True)
    if TARGET_BUILDER is not None:
        import annotation_targets
        config=json.dumps({'version':VERSION,'seed':SEED,'revision':revision,
                          'target_format':annotation_targets.TARGET_FORMAT,
                          'builder_sha256':sha(Path(annotation_targets.__file__).read_bytes()),
                          'split_source':'original-markdown-v1'},sort_keys=True)
    old=db.execute('SELECT value FROM config WHERE key="configuration"').fetchone()
    if old and old[0]!=config: raise ValueError('Preparation configuration changed; use a new output version')
    db.execute('INSERT OR IGNORE INTO config VALUES (?,?)',('configuration',config)); db.commit()
    return db

def pair_record(db,shard,stem,pair):
    group,augmented=group_id(shard,stem)
    def reject(reason,**details):
        if not augmented: db.execute('INSERT OR IGNORE INTO rejected_documents VALUES (?,?)',(group,reason))
        return {'id':shard+':'+stem,'group_id':group,'rejected':reason,
                'annotation_member':pair['json']['name'],'annotation_sha256':sha(pair['json']['data']),
                'image_member':pair['image']['name'],**details}
    rejected=db.execute('SELECT reason FROM rejected_documents WHERE id=?',(group,)).fetchone()
    if rejected:
        return reject('source_document_rejected: '+rejected[0])
    try:
        annotation=json.loads(pair['json']['data'])
    except (json.JSONDecodeError,UnicodeDecodeError) as exc:
        return reject('malformed_json_annotation',parse_error=str(exc))
    if not isinstance(annotation,dict):
        return reject('annotation_is_not_an_object')
    source_markdown=annotation.get('markdown')
    if not isinstance(source_markdown,str) or not source_markdown.strip():
        return reject('empty_or_missing_markdown')
    target_audit=None
    if TARGET_BUILDER is not None:
        from annotation_targets import TableTargetError
        try:
            resolved=TARGET_BUILDER(annotation)
        except TableTargetError as exc:
            return reject('invalid_table_target',target_error=str(exc))
        text=resolved.text; target_audit=resolved.audit
    else:
        text=source_markdown
    # Freeze v1 split membership even though the corrected label is now longer.
    # Hashing the new label here would move previously trained pages to validation.
    text_hash=sha(text.encode()); split=choose_split(source_markdown)
    augmentation=annotation.get('meta',{}).get('augmentation')
    if bool(augmentation)!=augmented: raise ValueError('Augmentation metadata and path disagree')
    original=db.execute('SELECT text_hash,split FROM documents WHERE id=?',(group,)).fetchone()
    if augmented:
        if not original or original!=(text_hash,split): raise ValueError('Augmented sample target differs from original: '+group)
    else:
        if original and original!=(text_hash,split): raise ValueError('Conflicting original document ID: '+group)
    image=pair['image']; data=image['data']
    try:
        with Image.open(io.BytesIO(data)) as im: im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load(); width,height=im.size; image_format=im.format
    except (OSError,SyntaxError) as exc:
        return reject('undecodable_image',image_sha256=sha(data),decode_error=str(exc))
    if width<=0 or height<=0: raise ValueError('Empty image dimensions')
    if not augmented: db.execute('INSERT OR IGNORE INTO documents VALUES (?,?,?)',(group,text_hash,split))
    image_hash=sha(data)
    duplicate=db.execute('SELECT text_hash,split FROM images WHERE hash=?',(image_hash,)).fetchone()
    if duplicate and duplicate!=(text_hash,split): raise ValueError('Duplicate image has conflicting label or split')
    db.execute('INSERT OR IGNORE INTO images VALUES (?,?,?)',(image_hash,text_hash,split))
    record={'id':shard+':'+stem,'group_id':group,'split':split,'text':text,'text_sha256':text_hash,
       'image':{'bucket':BUCKET,'key':'dataset/raw/'+shard,'member':image['name'],'offset':image['offset'],
                'size':len(data),'sha256':image_hash,'width':width,'height':height,'format':image_format},
       'annotation_member':pair['json']['name'],'augmented':augmented,'duplicate_image':bool(duplicate),'schema':VERSION}
    if target_audit is not None:
        record.update(target_format=target_audit['target_format'],table_audit=target_audit,
                      annotation_sha256=sha(pair['json']['data']),
                      split_source_sha256=sha(source_markdown.encode()),
                      language=annotation.get('meta',{}).get('language'))
    return record

def prepare_shard(s3,db,entry,work,update):
    shard=entry['path']; key='dataset/raw/'+shard
    if shutil.disk_usage(work).free<3*1024**3: raise RuntimeError('Less than 3 GiB free for preparation index and temporary manifest')
    temp=work/'current-manifest.jsonl'; rejected_temp=work/'current-rejections.jsonl'
    pending={}; pending_bytes=0; counts=Counter(); seen=set(); maximum=0
    response=s3.get_object(Bucket=BUCKET,Key=key); reader=DigestReader(response['Body'])
    try:
        with temp.open('w',encoding='utf-8') as output, rejected_temp.open('w',encoding='utf-8') as rejects, tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                if member.isdir(): continue
                if not member.isfile(): raise ValueError('Unsupported TAR entry type: '+member.name)
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe TAR member name')
                suffix=path.suffix.lower(); stem=str(path.with_suffix(''))
                role='json' if suffix=='.json' else 'image' if suffix in ('.png','.jpg','.jpeg') else None
                if role is None: raise ValueError('Unexpected file type: '+member.name)
                if member.size>32*1024**2: raise ValueError('Sample member exceeds 32 MiB memory budget')
                if stem in seen: raise ValueError('Repeated sample basename within a shard')
                pair=pending.setdefault(stem,{})
                if role in pair: raise ValueError('Duplicate image or JSON member')
                f=archive.extractfile(member); data=f.read(); f.close()
                if len(data)!=member.size: raise ValueError('Truncated TAR member')
                pair[role]={'name':member.name,'offset':member.offset_data,'data':data}
                pending_bytes+=len(data)
                if pending_bytes>128*1024**2: raise ValueError('Unpaired sample buffer exceeds 128 MiB')
                if len(pair)==2:
                    record=pair_record(db,shard,stem,pair)
                    counts['samples_seen']+=1
                    if 'rejected' in record:
                        rejects.write(json.dumps(record,ensure_ascii=False)+'\n'); counts['rejected']+=1
                    else:
                        output.write(json.dumps(record,ensure_ascii=False)+'\n')
                        counts[record['split']]+=1; counts['samples']+=1
                        counts['duplicate_images']+=int(record['duplicate_image'])
                        if 'table_audit' in record:
                            counts['samples_with_tables']+=int(record['table_audit']['table_count']>0)
                            counts['inline_tables']+=record['table_audit']['table_count']
                            counts['inline_cells']+=record['table_audit']['cell_count']
                        maximum=max(maximum,len(record['text']))
                    pending_bytes-=sum(len(v['data']) for v in pair.values())
                    del pending[stem]; seen.add(stem)
                    if counts['samples_seen']%1000==0:
                        output.flush()
                        update(shard=shard,shard_samples=counts['samples'])
            if pending: raise ValueError('Unmatched images or JSON annotations at end of shard')
            if not counts['samples']: raise ValueError('Shard has no samples')
            output.flush(); os.fsync(output.fileno())
            rejects.flush(); os.fsync(rejects.fileno())
        # TAR iteration stops at trailer; include remaining padding in its digest.
        while reader.read(8*1024*1024): pass
    finally: response['Body'].close()
    if reader.size!=entry['size'] or reader.hash.hexdigest()!=entry['lfs']['oid']:
        raise ValueError('Raw shard content changed or is truncated')
    manifest_key=f'dataset/processed/{VERSION}/{shard}.jsonl'
    manifest=upload_manifest(s3,temp,manifest_key)
    rejected_manifest=upload_manifest(s3,rejected_temp,f'dataset/manifests/{VERSION}/rejections/{shard}.jsonl')
    report={'shard':shard,'counts':dict(counts),'manifest':manifest,'rejections':rejected_manifest,'max_target_chars':maximum,'timestamp':now(),'raw_sha256':reader.hash.hexdigest()}
    db.execute('INSERT INTO shards VALUES (?,?)',(shard,json.dumps(report)))
    db.commit()
    temp.unlink() # Only task-owned derived temporary file, after verified upload.
    rejected_temp.unlink()
    return report

def main(*,version='ocr-manifest-v1',target_builder=None):
    global VERSION,TARGET_BUILDER
    if (version=='ocr-manifest-v1')!=(target_builder is None):
        raise ValueError('Legacy labels must stay in v1; corrected labels require a separate manifest version')
    VERSION=version; TARGET_BUILDER=target_builder
    parser=argparse.ArgumentParser(); parser.add_argument('--limit',type=int,default=0); args=parser.parse_args()
    evidence=ROOT/'evidence' if VERSION=='ocr-manifest-v1' else ROOT/'evidence'/VERSION
    evidence.mkdir(parents=True,exist_ok=True)
    lock=(evidence/'prepare.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    load_credentials(); s3=client()
    audit=json.loads((ROOT/'evidence/transfer-audit.json').read_text())
    if audit.get('status')!='passed': raise RuntimeError('Transfer audit must pass before preparation')
    work=ROOT/'preparation' if VERSION=='ocr-manifest-v1' else ROOT/'preparation'/VERSION
    work.mkdir(parents=True,exist_ok=True)
    db=create_db(work/'index.sqlite')
    tree=json.loads((ROOT/'evidence/dataset-tree.json').read_text())
    entries=sorted([e for e in tree if e['type']=='file' and e['path'].endswith('.tar')],key=lambda e:e['path'])
    state={'state':'running','version':VERSION,'total_shards':len(entries)}
    def update(**values):
        state.update(values); state['timestamp']=now()
        state['completed_shards']=db.execute('SELECT COUNT(*) FROM shards').fetchone()[0]
        atomic_json(evidence/'preparation-status.json',state)
    update()
    processed=0
    try:
        for entry in entries:
            old=db.execute('SELECT report FROM shards WHERE path=?',(entry['path'],)).fetchone()
            if old:
                saved=json.loads(old[0]); m=saved['manifest']; head=s3.head_object(Bucket=BUCKET,Key=m['key'])
                if head['ContentLength']!=m['bytes'] or head.get('Metadata',{}).get('sha256')!=m['sha256']:
                    raise ValueError('Previously prepared manifest changed')
                continue
            update(shard=entry['path'],shard_samples=0)
            db.execute('BEGIN')
            report=prepare_shard(s3,db,entry,work,update)
            print(json.dumps(report),flush=True)
            put_verified(s3,BUCKET,f'dataset/manifests/{VERSION}/{entry["path"]}.json',json.dumps(report).encode())
            update()
            status_key='dataset/manifests/preparation-status.json' if VERSION=='ocr-manifest-v1' else f'dataset/manifests/{VERSION}/preparation-status.json'
            put_verified(s3,BUCKET,status_key,json.dumps(state).encode())
            processed+=1
            if args.limit and processed>=args.limit: break
        reports=[json.loads(r[0]) for r in db.execute('SELECT report FROM shards ORDER BY path')]
        totals=Counter()
        for report in reports: totals.update(report['counts'])
        complete=len(reports)==len(entries)
        update(state='complete' if complete else 'partial',counts=dict(totals))
        if complete:
            split_method='SHA256(seed + normalized markdown), source-group validation and cross-split image hash checks' if TARGET_BUILDER is None else 'SHA256(seed + original normalized markdown), source-group validation and cross-split image hash checks'
            result={'schema':VERSION,'seed':SEED,'split_method':split_method,
                    'counts':dict(totals),'shards':reports,'timestamp':now()}
            if TARGET_BUILDER is not None:
                from annotation_targets import TARGET_FORMAT
                result.update(target_format=TARGET_FORMAT,split_membership='Preserved from v1 original Markdown; never rehash the resolved target.')
            atomic_json(evidence/'preparation-summary.json',result)
            put_verified(s3,BUCKET,f'dataset/manifests/{VERSION}/COMPLETE.json',json.dumps(result).encode())
        status_key='dataset/manifests/preparation-status.json' if VERSION=='ocr-manifest-v1' else f'dataset/manifests/{VERSION}/preparation-status.json'
        put_verified(s3,BUCKET,status_key,json.dumps(state).encode())
    except Exception as exc:
        db.rollback(); update(state='failed',error_type=type(exc).__name__,error=str(exc))
        print(json.dumps(state),flush=True)
        status_key='dataset/manifests/preparation-status.json' if VERSION=='ocr-manifest-v1' else f'dataset/manifests/{VERSION}/preparation-status.json'
        try: put_verified(s3,BUCKET,status_key,json.dumps(state).encode())
        except Exception: pass
        raise
    finally: db.close()

if __name__=='__main__': main()
