"""Project-owned runtime helpers; credentials stay in process memory."""
import base64, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps'))

def load_credentials():
    if all(os.getenv(k) for k in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_ENDPOINT_URL')):
        return
    result=subprocess.run(['oc','get','secret','q35ocr-20260921-p560-s3','-n','mtrna-ns1','-o','json'],capture_output=True,text=True,check=True)
    secret=json.loads(result.stdout)
    if secret['metadata'].get('labels',{}).get('project')!='q35ocr-20260921-p560':
        raise RuntimeError('Secret ownership mismatch')
    for k in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_ENDPOINT_URL'):
        os.environ[k]=base64.b64decode(secret['data'][k]).decode()

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)

def put_verified(s3,bucket,key,data):
    digest=hashlib.sha256(data).hexdigest()
    s3.put_object(Bucket=bucket,Key=key,Body=data,Metadata={'sha256':digest,'project':'q35ocr-20260921-p560'})
    response=s3.get_object(Bucket=bucket,Key=key)
    try:
        actual=hashlib.sha256(response['Body'].read()).hexdigest()
    finally:
        response['Body'].close()
    if actual!=digest: raise RuntimeError('Uploaded object checksum mismatch: '+key)
    return digest
