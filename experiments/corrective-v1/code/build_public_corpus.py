"""Prepare a small balanced corrective corpus; old evaluation documents excluded."""
import concurrent.futures,hashlib,io,json,lzma,os,random,re,tarfile,unicodedata,urllib.parse,urllib.request
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent;CACHE=ROOT/'public-sources';OUT=ROOT/os.environ.get('OCR_CORPUS_DIRECTORY','corrective-corpus-v1')
OUT.mkdir(exist_ok=True)
for n in ('images','original-images','references','metadata'):(OUT/n).mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
BENCH=ROOT.parent/'benchmark-31-20261026/livrables/comparatif-31-images'
held=json.loads((BENCH/'jeu.json').read_text(encoding='utf-8'))
held_docs={c['dataset_id'].split('_page_')[0] for c in held['cases']}
held_images={c['inference_image_sha256'] for c in held['cases']}
revision='2684870ad1a0ba271ed56c31e19cb05a6b78c291'
def get(url):
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OCR-fidelity-research/1.0'}),timeout=50) as r:return r.read()
 except Exception as e:raise RuntimeError('Public data request failed: '+type(e).__name__) from None
def save_case(case_id,lang,split,doc,image_raw,text,source,source_id,license):
 ref=OUT/'references'/(case_id+'.md');ref.write_text(text,encoding='utf-8')
 with Image.open(io.BytesIO(image_raw)) as im:original=im.convert('RGB')
 original_path=OUT/'original-images'/(case_id+'.png')
 if not original_path.exists():original.save(original_path)
 image=original.copy();image.thumbnail((1024,1024));path=OUT/'images'/(case_id+'.png');image.save(path)
 digest=sha(path.read_bytes());assert digest not in held_images
 return {'id':case_id,'language':lang,'split':split,'document_id':doc,'image':'images/'+case_id+'.png','image_sha256':digest,
  'original_image':'original-images/'+case_id+'.png','original_image_sha256':sha(original_path.read_bytes()),
  'reference':'references/'+case_id+'.md','text_sha256':sha(ref.read_bytes()),'source':source,'source_id':source_id,'license':license,
  'annotation_review':'publisher annotation; independent complete line-by-line review pending','synthetic_control':False}
records=[];quarantine=[{'source':'Nayana','document':'en_pdf_0170','reason':'visual review: visible Cyrillic stamp and handwritten date missing from publisher target; entire document excluded'}]
def image_info(row):
 value=row['image']
 return value[0] if isinstance(value,list) else value
def agrees_with_language(text,lang):
 letters=[c for c in text if unicodedata.category(c).startswith('L')]
 latin=sum('LATIN' in unicodedata.name(c,'') for c in letters)
 if not letters or latin/len(letters)<.85:return False
 words=re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿœŒ]+",text.lower())
 en=set('the and of to in is for that with as on by are from this be was which at an or it have has can will were their these they not but been we our its into than such'.split())
 fr=set('le la les de des du un une et est en pour que qui dans sur au aux par avec ce cette ces sont se ne pas nous vous il elle ils elles dont plus ou être ont été'.split())
 a=sum(w in en for w in words);b=sum(w in fr for w in words)
 return a>=max(3,len(words)*.03) and a>b*1.1 if lang=='en' else b>=max(5,len(words)*.04) and b>a*1.2
# NOD articles are distinct documents; use clean colour scans only.
gt={}
with lzma.open(CACHE/'ground_truth.tar.lzma','rb') as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
 for m in archive:
  if m.isfile() and 'yarmouk' in m.name.lower():
   raw=archive.extractfile(m).read()
   text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
   gt[Path(m.name).stem]=text.replace('\r\n','\n').strip()+'\n'
print('NOD_REFERENCES',len(gt),'KEY_EXAMPLES',list(gt)[:5],flush=True)
members=json.loads((CACHE/'yarmouk_01_col.tar.lzma.members.json').read_text())
names=sorted(Path(x['name']).stem for x in members);random.Random(20261027).shuffle(names)
assignment={name:'train' if i<80 else 'validation' for i,name in enumerate(names)};assert len(assignment)==100
if all((OUT/'original-images'/('nod-ar-'+name+'.png')).is_file() for name in names):
 for item in members:
  name=Path(item['name']).stem;key=name if name in gt else name.removesuffix('_1')
  if '\ufffd' in gt[key]:
   quarantine.append({'source':'NOD','document':key,'reason':'replacement characters already present in publisher transcription'});continue
  records.append(save_case('nod-ar-'+name,'ar',assignment[name],'NOD:Yarmouk:'+key,(OUT/'original-images'/('nod-ar-'+name+'.png')).read_bytes(),gt[key],
   'https://zenodo.org/records/5068735',item['name'],'CC-BY-4.0'))
 print('NOD_EXISTING_VERIFIED_ARCHIVE_IMAGES_REUSED',len(records),flush=True)
else:
 with lzma.open(CACHE/'yarmouk_01_col.tar.lzma','rb') as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
  for m in archive:
   if not m.isfile():continue
   name=Path(m.name).stem;key=name if name in gt else name.removesuffix('_1')
   assert key in gt,('Missing article reference',name)
   if '\ufffd' in gt[key]:
    quarantine.append({'source':'NOD','document':key,'reason':'replacement characters already present in publisher transcription'});continue
   records.append(save_case('nod-ar-'+name,'ar',assignment[name],'NOD:Yarmouk:'+key,archive.extractfile(m).read(),gt[key],
    'https://zenodo.org/records/5068735',m.name,'CC-BY-4.0'))
   if len(records)%20==0:print('NOD_SCANS_PREPARED',len(records),flush=True)
# Refresh only metadata; do not rely on expired signed image links from old runs.
hub=CACHE/'nayana-current-meta.json'
if not hub.exists():hub.write_bytes(get('https://huggingface.co/api/datasets/Cognitive-Lab/NayanaOCRBench_Natural'))
assert json.loads(hub.read_text())['sha']==revision,'Public corpus revision changed; do not silently mix releases'
tasks=[]
for lang in ('fr','en'):
 for offset in range(0,600,100):tasks.append((lang,offset))
def metadata(task):
 lang,offset=task;p=CACHE/f'nayana-fresh-{lang}-{offset:03d}.json'
 if not p.exists() or os.environ.get('OCR_REFRESH_METADATA')=='1':p.write_bytes(get('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':'Cognitive-Lab/NayanaOCRBench_Natural','config':lang,'split':'train','offset':offset,'length':100})))
 return lang,json.loads(p.read_text(encoding='utf-8'))['rows']
allrows=defaultdict(list)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for lang,rows in pool.map(metadata,tasks):allrows[lang].extend(rows)
for lang in ('fr','en'):
 groups=defaultdict(list)
 suspect_docs=set()
 for row in allrows[lang]:
  c=row['row'];parts=[re.sub(r'\s+',' ',x).strip() for x in c['md_norm1000'].split('\n\n') if len(x.strip())>100]
  if any(n>1 for n in Counter(parts).values()):suspect_docs.add(c['id'].split('_page_')[0])
 quarantine.extend({'source':'Nayana','document':d,'reason':'repeated long publisher target paragraph; entire document quarantined rather than automatic deduplication'} for d in sorted(suspect_docs))
 for row in allrows[lang]:
  c=row['row'];doc=c['id'].split('_page_')[0];text=c['md_norm1000']
  if doc in held_docs or doc in suspect_docs or row['truncated_cells'] or not 350<=len(text)<=6000:continue
  if '\ufffd' in text:continue
  if not agrees_with_language(text,lang):continue
  if re.search(r'curriculum vitae|\bcv\b|résumé professionnel',text,re.I):continue
  if c['id']=='en_pdf_0009_page_00001':continue
  # Visual review found an untranscribed Cyrillic stamp/date in this document.
  # Exclude the whole document rather than teach omission of visible text.
  if doc=='en_pdf_0170':continue
  image=image_info(c);area=image['height']*image['width']
  if area>18_000_000:continue
  groups[doc].append(c)
 documents=sorted(groups);random.Random(20261027+(lang=='fr')).shuffle(documents)
 val_docs=set(documents[:max(15,len(documents)//5)]);train_docs=set(documents)-val_docs
 assert not val_docs & train_docs and not (val_docs|train_docs) & held_docs
 selected=[]
 for split,docs,total in [('train',train_docs,80),('validation',val_docs,20)]:
  candidates=[]
  for doc in sorted(docs):
   pages=groups[doc][:];random.Random(doc).shuffle(pages)
   for i,c in enumerate(pages):candidates.append((i,doc,c))
  candidates.sort(key=lambda x:(x[0],sha(x[1].encode())))
  assert len(candidates)>=total,(lang,split,len(candidates))
  for _,doc,c in candidates[:total]:selected.append((split,doc,c))
 def download(item):
  split,doc,c=item;case_id='nayana-'+c['id'];p=CACHE/(case_id+'.source-image')
  if not p.exists():p.write_bytes(get(image_info(c)['src']))
  text=re.sub(r'<!--.*?-->','',c['md_norm1000'],flags=re.S)
  text=re.sub(r'!\[[^\]]*\]\([^)]*\)','',text).strip()+'\n'
  return save_case(case_id,lang,split,'Nayana:'+doc,p.read_bytes(),text,
   'https://huggingface.co/datasets/Cognitive-Lab/NayanaOCRBench_Natural/tree/'+revision,c['id'],'CC-BY-NC-4.0')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(download,selected),1):
   records.append(r)
   if i%20==0:print('NAYANA_IMAGES_PREPARED',lang,i,flush=True)
counts=Counter((r['split'],r['language']) for r in records)
assert all(counts[('train',lang)]>=75 and counts[('validation',lang)]>=18 for lang in ('ar','fr','en'))
assert len({r['image_sha256'] for r in records})==len(records),'Duplicate image in corrective corpus'
split_docs={split:{r['document_id'] for r in records if r['split']==split} for split in ('train','validation')};assert not split_docs['train'] & split_docs['validation']
(OUT/'real-pages.json').write_text(json.dumps({'cases':records,'old_31_evaluation_documents_excluded':True,'split_unit':'document/article','all_references_independently_verified':False},ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'metadata/quarantine.json').write_text(json.dumps(quarantine,indent=2))
print('PUBLIC_REAL_CORPUS_PREPARED',json.dumps({str(k):v for k,v in counts.items()}),flush=True)
