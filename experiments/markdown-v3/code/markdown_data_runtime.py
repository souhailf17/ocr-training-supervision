"""Shared verified Markdown encoding and bounded MinIO I/O for corpus v3."""
import hashlib,io,json,os,time
from pathlib import Path
from markdown_table_targets import TARGET_FORMAT,PROMPT,sha
from annotation_targets import assert_no_table_placeholders

BUCKET='qwen35-9b-ocr-finetuning'
SOURCE_SHA='89e75461a805005be53d18e772df31422f06a8fc9f3bbd32f636350cc5bd153b'
PREFIX='dataset/processed/ocr-markdown-v3-orig-aug1/'
MAX_SEQUENCE=8192

def s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',
        config=Config(connect_timeout=10,read_timeout=90,max_pool_connections=20,
                      retries={'max_attempts':0},s3={'addressing_style':'path'}))

def retry(operation):
    from minio_retry import retryable,RETRY_DELAYS
    for attempt in range(4):
        try:return operation()
        except Exception as exc:
            if not retryable(exc) or attempt==3:raise
            print('MARKDOWN_IO_RETRY',json.dumps({'attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
            time.sleep(RETRY_DELAYS[attempt])

def read(s3,key):
    from minio_retry import read_object
    return read_object(s3,BUCKET,key)

def publish(s3,key,body):
    if not key.startswith(PREFIX):raise ValueError('Writes restricted to the new corpus prefix')
    digest=sha(body)
    def put():
        s3.put_object(Bucket=BUCKET,Key=key,Body=body,Metadata={'sha256':digest,'target-format':TARGET_FORMAT})
        if read(s3,key)!=body:raise ValueError('Uploaded object readback differs: '+key)
    retry(put)
    return {'key':key,'sha256':digest,'bytes':len(body)}

def encode(processor,row,raw,pin=False,allow_oversize=False):
    import torch
    from PIL import Image
    if row['target_format']!=TARGET_FORMAT:raise ValueError('Wrong target version')
    assert_no_table_placeholders(row['text'])
    if sha(raw)!=row['image']['sha256'] or sha(row['text'])!=row['text_sha256']:
        raise ValueError('Input/target digest mismatch')
    with Image.open(io.BytesIO(raw)) as im:image=im.convert('RGB')
    image.thumbnail((1024,1024))
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
    kwargs=dict(tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    prompt=processor.apply_chat_template(messages,add_generation_prompt=True,**kwargs)
    encoded=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':row['text']}]}],add_generation_prompt=False,**kwargs)
    n=prompt['input_ids'].shape[1]
    if not torch.equal(prompt['input_ids'],encoded['input_ids'][:,:n]):raise ValueError('Assistant prefix mismatch')
    if encoded['input_ids'].shape[1]>MAX_SEQUENCE and not allow_oversize:
        raise ValueError('Sequence exceeds unchanged 8192-token limit; no truncation')
    labels=encoded['input_ids'].clone();labels[:,:n]=-100;encoded['labels']=labels
    if int((labels[:,1:]!=-100).sum())<1:raise ValueError('Empty supervised target')
    if pin:encoded={k:v.pin_memory() for k,v in encoded.items()}
    return encoded

def check_table_roundtrip(resolved):
    from markdown_it import MarkdownIt
    from annotation_targets import parse_tables
    parser=MarkdownIt('commonmark',{'html':True}).enable('table')
    for audit in resolved.audit['tables']:
        # Reconstruct just this table from the audited anchors, then render it
        # with the actual independent GFM parser and compare every occupied slot.
        from markdown_table_targets import escape,visible
        grid=[['']*audit['columns'] for _ in range(audit['rows'])]
        for a in audit['anchors']:grid[a['row']][a['column']]=a['text']
        if audit['synthetic_empty_header']:grid.insert(0,['']*audit['columns'])
        lines=['| '+' | '.join(escape(c) for c in grid[0])+' |',
               '| '+' | '.join(['---']*audit['columns'])+' |']
        lines+=['| '+' | '.join(escape(c) for c in row)+' |' for row in grid[1:]]
        parsed=parse_tables(parser.render('\n'.join(lines)))
        got=[[visible(c['text']) for c in r] for r in parsed[0]['rows']] if len(parsed)==1 else None
        if got!=grid:raise ValueError('GFM roundtrip changed a cell')

def selected_sources(complete):
    import re
    entries=[e for e in complete['shards'] if re.fullmatch(r'data/shard_\d+\.tar',e['shard'])
             or re.fullmatch(r'data_aug/shard_\d+_aug1\.tar',e['shard'])]
    by={e['shard']:e for e in entries};pairs=[]
    for n in range(1,104):
        a=f'data/shard_{n:03d}.tar';b=f'data_aug/shard_{n:03d}_aug1.tar'
        if a not in by or b not in by:raise ValueError('Original/aug1 source family missing')
        pairs.append((by[a],by[b]))
    if len(entries)!=206:raise ValueError('Expected all 103 original + 103 aug1 shards')
    return pairs

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
