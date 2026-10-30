import hashlib,io,json,os,tarfile,unicodedata
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/os.environ.get('OCR_CORPUS_DIRECTORY','corrective-corpus-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=json.loads((OUT/'real-pages.json').read_text(encoding='utf-8'));cases=data['cases'][:]
for spec in json.loads((OUT/'metadata/control-specs.json').read_text(encoding='utf-8')):
 p=OUT/'images'/(spec['id']+'.png');assert p.exists()
 cases.append({k:spec[k] for k in ('id','language','split','reference','synthetic_control')}|{'image':'images/'+p.name,'image_sha256':sha(p),
  'text_sha256':sha(OUT/spec['reference']),'document_id':spec['id'],'source':'deterministic-HTML-glyph-control','source_id':spec['id'],
  'license':'CC0-1.0','annotation_review':'literal source text matched rendered DOM; visual glyph review required','digit_family':spec['digit_family']})
cases+=json.loads((OUT/'metadata/blank-controls.json').read_text())
counts=Counter();digit_counts=Counter();docs={'train':set(),'validation':set()};images=set()
for c in cases:
 assert sha(OUT/c['image'])==c['image_sha256'] and sha(OUT/c['reference'])==c['text_sha256']
 assert c['image_sha256'] not in images;images.add(c['image_sha256']);counts[(c['split'],c['language'])]+=1;docs[c['split']].add(c['document_id'])
 text=(OUT/c['reference']).read_text(encoding='utf-8')
 assert '\ufffd' not in text and '<|im_end|>' not in text and '<|im_start|>' not in text
 c['prompt_language']='mixed' if c['language'] in ('fr','en') and any(unicodedata.category(x).startswith('L') and 'ARABIC' in unicodedata.name(x,'') for x in text) else c['language']
 for family,chars in [('latin','0123456789'),('arabic_indic','٠١٢٣٤٥٦٧٨٩'),('eastern_arabic_indic','۰۱۲۳۴۵۶۷۸۹')]:digit_counts[family]+=sum(x in chars for x in text)
assert not docs['train']&docs['validation']
assert len(cases)==375 and all(counts[('train',lang)]==100 and counts[('validation',lang)]==25 for lang in ('ar','fr','en'))
assert digit_counts['arabic_indic']>400 and digit_counts['eastern_arabic_indic']>400
manifest={'version':'literal-fidelity-public-v1','counts':{str(k):v for k,v in counts.items()},'cases':cases,
 'digit_codepoint_counts':dict(digit_counts),'old31_document_groups_excluded':True,'split_unit':'document/article for real pages; separately generated controls',
 'all_annotations_independently_retranscribed':False,'real_pages':len(data['cases']),'synthetic_literal_controls':len(cases)-len(data['cases'])-15,'synthetic_blank_controls':15,
 'source_rights':{'NOD':'CC-BY-4.0; Thomas Hegghammer (2021), doi:10.5281/zenodo.5068735','Nayana':'CC-BY-NC-4.0; Cognitive-Lab/NayanaOCRBench_Natural @2684870ad1a0ba271ed56c31e19cb05a6b78c291'},
 'preprocessing':'RGB; PIL thumbnail((1024,1024)) BICUBIC; no glyph-family or punctuation normalization',
 'training_ready':False,'pending_gates':['visual sample review','CPU encoder and EOS/length checks','generation validation before model promotion']}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=ROOT/(OUT.name+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 t.add(OUT/'manifest.json',arcname='manifest.json')
 for c in cases:
  for key in ('image','reference'):t.add(OUT/c[key],arcname=c[key])
print('CORRECTIVE_CORPUS_FROZEN',json.dumps({'cases':len(cases),'counts':manifest['counts'],'digit_glyphs':dict(digit_counts),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)}),flush=True)
