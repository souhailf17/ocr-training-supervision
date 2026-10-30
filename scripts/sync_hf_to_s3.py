"""Pinned HF -> bounded-memory multipart S3 transfer with full read-back verification.

Only one object transfers at a time. Never stores raw shards on disk. Retries
restart the current object, while completed objects are resumable. No ETag is
assumed to be an MD5 hash. Dataset access requires an approved HF token.
"""
import argparse, datetime, hashlib, json, os, re, time, urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse
from botocore.exceptions import ClientError
from s3_common import client, BUCKET, PROJECT

PART = 32 * 1024 * 1024

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new and urlparse(newurl).hostname != 'huggingface.co':
            new.remove_header('Authorization')
        return new

def verify(s3, key, size, digest):
    head = s3.head_object(Bucket=BUCKET, Key=key)
    if head['ContentLength'] != size:
        raise ValueError('S3 size mismatch: '+key)
    result = s3.get_object(Bucket=BUCKET, Key=key)
    h = hashlib.sha256()
    count = 0
    try:
        for chunk in result['Body'].iter_chunks(chunk_size=PART):
            h.update(chunk)
            count += len(chunk)
    finally:
        result['Body'].close()
    if count != size or h.hexdigest() != digest:
        raise ValueError('S3 read-back checksum mismatch: '+key)

def event(s3, log, record):
    record['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    line = json.dumps(record, ensure_ascii=False)
    with log.open('a', encoding='utf-8') as f:
        f.write(line+'\n')
        f.flush()
        os.fsync(f.fileno())
    ident = hashlib.sha256(record['source_path'].encode()).hexdigest()
    s3.put_object(Bucket=BUCKET, Key=f'dataset/manifests/transfers/{record["kind"]}/{ident}.json', Body=line.encode(), ContentType='application/json')
    print(line, flush=True)

def transfer(s3, opener, token, kind, summary, item, log):
    prefix = 'model/base' if kind == 'model' else 'dataset/raw'
    key = prefix+'/'+item['path']
    hf_prefix = '' if kind == 'model' else 'datasets/'
    url = f'https://huggingface.co/{hf_prefix}{summary["repository"]}/resolve/{summary["revision"]}/{quote(item["path"], safe="/")}'
    expected = item.get('lfs', {}).get('oid')
    if expected and not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('Source checksum is redacted or invalid; refresh the authenticated pinned inventory before transferring')
    metadata = {'project': PROJECT, 'hf-revision': summary['revision'], 'hf-path': item['path']}
    if expected:
        metadata['hf-sha256'] = expected
    record = dict(kind=kind, source_url=url, source_path=item['path'], s3_bucket=BUCKET, s3_object=key, file_size=item['size'])
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError as e:
        if e.response['Error']['Code'] not in ('404','NoSuchKey','NotFound'):
            raise
        head = None
    if head:
        if any(head.get('Metadata',{}).get(k) != v for k,v in metadata.items()):
            raise RuntimeError('Refusing to overwrite an object with different provenance: '+key)
        digest = expected
        if not digest:
            ident = hashlib.sha256(item['path'].encode()).hexdigest()
            try:
                saved = json.loads(s3.get_object(Bucket=BUCKET, Key=f'dataset/manifests/transfers/{kind}/{ident}.json')['Body'].read())
                if saved.get('source_url') == url and saved.get('status') in ('verified','verified-existing'):
                    digest = saved['checksum']
            except ClientError as e:
                if e.response['Error']['Code'] not in ('404','NoSuchKey','NotFound'):
                    raise
        if digest:
            verify(s3,key,item['size'],digest)
            event(s3,log,dict(record,checksum=digest,status='verified-existing'))
            return
    for attempt in range(1,5):
        upload_id = None
        try:
            headers = {'User-Agent':'q35ocr-transfer/1.0'}
            if token:
                headers['Authorization'] = 'Bearer '+token
            sha = hashlib.sha256()
            git = hashlib.sha1(f'blob {item["size"]}\0'.encode())
            parts, count = [], 0
            with opener.open(urllib.request.Request(url, headers=headers),timeout=120) as response:
                upload_id = s3.create_multipart_upload(Bucket=BUCKET,Key=key,Metadata=metadata)['UploadId']
                while True:
                    data = response.read(PART)
                    if not data:
                        break
                    sha.update(data); git.update(data); count += len(data)
                    part = s3.upload_part(Bucket=BUCKET,Key=key,UploadId=upload_id,PartNumber=len(parts)+1,Body=data)
                    parts.append({'ETag':part['ETag'],'PartNumber':len(parts)+1})
                    if len(parts) % 8 == 0:
                        print(json.dumps({'event':'progress','object':key,'bytes':count,'total':item['size']}),flush=True)
            digest=sha.hexdigest()
            if count != item['size'] or (expected and digest != expected):
                raise ValueError('Source size/checksum mismatch')
            if not expected and item.get('oid') and git.hexdigest() != item['oid']:
                raise ValueError('Source Git blob checksum mismatch')
            if not parts:
                raise ValueError('Unexpected empty source object')
            s3.complete_multipart_upload(Bucket=BUCKET,Key=key,UploadId=upload_id,MultipartUpload={'Parts':parts})
            upload_id=None
            verify(s3,key,count,digest)
            # Preserve the verified digest in a separate manifest; no costly
            # server-side copy of multi-GB objects just to change metadata.
            event(s3,log,dict(record,checksum=digest,status='verified',attempt=attempt))
            return
        except Exception as exc:
            if upload_id:
                s3.abort_multipart_upload(Bucket=BUCKET,Key=key,UploadId=upload_id)
            event(s3,log,dict(record,checksum=None,status='failed',attempt=attempt,error_type=type(exc).__name__))
            if attempt == 4:
                raise
            time.sleep(min(2**attempt,30))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['model','dataset'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    s3=client()
    owner=json.loads(s3.get_object(Bucket=BUCKET,Key='documentation/project-owner.json')['Body'].read())
    if owner.get('project') != PROJECT:
        raise RuntimeError('Project ownership check failed')
    summary=json.loads((args.root/'evidence/source-summary.json').read_text())[args.kind]
    entries=json.loads((args.root/f'evidence/{args.kind}-tree.json').read_text())
    entries=[e for e in entries if e['type']=='file']
    if args.limit:
        entries=entries[:args.limit]
    token=os.getenv('HF_TOKEN')
    tokenpath=Path.home()/'.cache/huggingface/token'
    if not token and tokenpath.exists():
        token=tokenpath.read_text().strip()
    log=args.root/f'evidence/{args.kind}-transfer.jsonl'
    opener=urllib.request.build_opener(SafeRedirect())
    for item in entries:
        transfer(s3,opener,token,args.kind,summary,item,log)

if __name__=='__main__':
    main()
