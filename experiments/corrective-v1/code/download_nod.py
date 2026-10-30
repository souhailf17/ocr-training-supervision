import hashlib,json,lzma,tarfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CACHE=ROOT/'public-sources';meta=json.loads((CACHE/'nod-record.json').read_text())
for f in meta['files']:
 if f['key'] not in ('ground_truth.tar.lzma','yarmouk_01_col.tar.lzma'):continue
 p=CACHE/f['key'];expected=f['checksum'].split(':')[1]
 if p.exists():
  with p.open('rb') as h:assert hashlib.file_digest(h,'md5').hexdigest()==expected
 else:
  target=p.with_suffix(p.suffix+'.partial');digest=hashlib.md5();size=0
  try:
   with urllib.request.urlopen(urllib.request.Request(f['links']['self'],headers={'User-Agent':'OCR-fidelity-research/1.0'}),timeout=50) as response,target.open('wb') as out:
    while True:
     block=response.read(1024*1024)
     if not block:break
     out.write(block);digest.update(block);size+=len(block)
  except Exception as e:raise RuntimeError('Public download failed: '+p.name+' '+type(e).__name__) from None
  assert size==f['size'] and digest.hexdigest()==expected;target.replace(p)
 print('DOWNLOADED_AND_VERIFIED',p.name,p.stat().st_size,flush=True)
 members=[]
 with lzma.open(p,'rb') as source,tarfile.open(fileobj=source,mode='r|') as archive:
  for m in archive:
   if m.isfile():members.append({'name':m.name,'bytes':m.size})
 (CACHE/(p.name+'.members.json')).write_text(json.dumps(members,indent=2))
 print('FILES',len(members),'EXAMPLES',members[:8],flush=True)
