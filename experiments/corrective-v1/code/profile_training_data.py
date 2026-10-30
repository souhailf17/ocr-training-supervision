"""Sampled script/digit distribution and actual supervised EOS audit."""
import hashlib,io,json,os,unicodedata
from pathlib import Path
from collections import Counter
from minio_retry import read_object
import boto3
from botocore.config import Config
ROOT=Path('/export/ocr-fidelity-20261027-v1');TRAIN=Path('/work/q35ocr-20260921-p560/training-markdown-v3-orig-aug1')
s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',config=Config(connect_timeout=10,read_timeout=60,retries={'max_attempts':2},s3={'addressing_style':'path'}))
complete=json.loads((TRAIN/'data/COMPLETE.json').read_text());counts=Counter();digits=Counter();selected=[];summaries=[]
for index in (0,51,102,153,205):
 m=complete['shards'][index]['manifest'];raw=read_object(s3,'qwen35-9b-ocr-finetuning',m['key']);assert hashlib.sha256(raw).hexdigest()==m['sha256']
 rows=[json.loads(line) for line in raw.splitlines() if line.strip()];train=[r for r in rows if r['split']=='train']
 script=Counter()
 for r in train:
  a=l=0
  for c in r['text']:
   name=unicodedata.name(c,'')
   if unicodedata.category(c).startswith('L'):
    a+='ARABIC' in name;l+='LATIN' in name
   for family,chars in [('latin','0123456789'),('arabic_indic','٠١٢٣٤٥٦٧٨٩'),('eastern_arabic_indic','۰۱۲۳۴۵۶۷۸۹')]:
    if c in chars:digits[family]+=1
  key='arabic_dominant' if a>l else 'latin_dominant' if l>a else 'other_or_equal'
  counts[key]+=1;script[key]+=1;counts['pages']+=1
  if any(c in r['text'] for c in 'éèêàçœÉÀ'):counts['pages_with_french_accent_clue']+=1
 selected.extend(train[:1]);summaries.append({'shard':index,'pages':len(train),'scripts':dict(script)})
 print('SCRIPT_PROFILE_SHARD',index,json.dumps(dict(script)),flush=True)
from transformers import AutoProcessor
from markdown_data_runtime import encode
processor=AutoProcessor.from_pretrained('/export/markdown-best-001442/base-reference',local_files_only=True)
end=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');eos=[]
for row in selected:
 image=row['image'];kwargs={} if image.get('offset') is None else {'byte_range':f"bytes={image['offset']}-{image['offset']+image['size']-1}"}
 raw=read_object(s3,'qwen35-9b-ocr-finetuning',image['key'],**kwargs);batch=encode(processor,row,raw)
 labels=batch['labels'][0].tolist();positions=[i for i,v in enumerate(labels) if v==end]
 eos.append({'id':row.get('id'),'supervised_im_end_count':len(positions),'last_supervised_token_ids':[v for v in labels if v!=-100][-8:]})
 assert positions,'Missing supervised end-of-assistant token'
payload={'sampled_manifests':5,'total_manifests':206,'counts':dict(counts),'digit_codepoints':dict(digits),'shards':summaries,'eos_examples':eos,'language_method':'Unicode script character heuristic, not a French/English language classifier','sample_not_complete_corpus':True}
(ROOT/'data-profile.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));print('TRAINING_DATA_PROFILE',json.dumps(payload),flush=True)
