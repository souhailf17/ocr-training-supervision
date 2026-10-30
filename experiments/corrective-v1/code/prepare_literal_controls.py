"""Deterministic synthetic glyph/EOS controls, clearly separate from real scans."""
import hashlib,json,os,random
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent;OUT=ROOT/os.environ.get('OCR_CORPUS_DIRECTORY','corrective-corpus-v1');specs=[]
alphabets=['0123456789','٠١٢٣٤٥٦٧٨٩','۰۱۲۳۴۵۶۷۸۹']
real=json.loads((OUT/'real-pages.json').read_text())['cases']
for lang in ('ar','fr','en'):
 for split,target,blanks in (('train',100,4),('validation',25,1)):
  count=target-blanks-sum(c['language']==lang and c['split']==split for c in real)
  for i in range(count):
   case=f'control-{lang}-{split}-{i:02d}';rng=random.Random(case);family=i%3
   def number(value):return str(value).translate(str.maketrans(alphabets[0],alphabets[family]))
   code=number(rng.randrange(10000,99999));amount=number(rng.randrange(110,9999));fraction=number(rng.randrange(10,99))
   if lang=='fr':
    title=['Relevé de mesures','Extrait de registre','Procès-verbal','Copie fidèle'][i%4]
    lines=[f'Référence : FR-{code}',f'Montant : {amount},{fraction} €',f'Échéance : {number(rng.randrange(1,28))}/09/{number(2026)}',
     'Caractères : œ, æ, ç, é, è, ê, É, À.', 'Citation : « Déjà réglé » ; n° 5.',f'Taux : {number(rng.randrange(1,99))} %','Mesures : ± ≤ ≥ × ÷ − ; m².', 'Suite de chiffres : '+number(rng.randrange(10**19,10**20-1))]
   elif lang=='en':
    title=['Measurement record','Register extract','Audit sheet','Document copy'][i%4]
    lines=[f'Reference: EN-{code}',f'Amount: € {amount}.{fraction}',f'Date: {number(rng.randrange(1,28))}/09/{number(2026)}',
     'Characters: œ, æ, ç, é, è, ê, É, À.', 'Quote: “Exact copy” — item № 5.',f'Rate: {number(rng.randrange(1,99))} %','Measurements: ± ≤ ≥ × ÷ − ; m².', 'Digit string: '+number(rng.randrange(10**19,10**20-1))]
   else:
    title=['سجل القياسات','بيان إداري','نسخة مطابقة','إفادة'][i%4]
    lines=[f'الرقم المرجعي: AR-{code}',f'القيمة: {amount}٫{fraction} د.م.',f'التاريخ: {number(rng.randrange(1,28))}/09/{number(2026)}',
     'البيان: نَصٌّ عَرَبِيٌّ بِالتَّشْكِيلِ.', 'الرموز: « » ، ؛ ؟ ﴿ ﴾ ﷺ.',f'النسبة: {number(rng.randrange(1,99))} %','القياسات: ± ≤ ≥ × ÷ − ; m².', 'سلسلة أرقام: '+number(rng.randrange(10**19,10**20-1))]
   text='# '+title+'\n\n'+'\n\n'.join(lines)+'\n'
   ref=OUT/'references'/(case+'.md');ref.write_text(text,encoding='utf-8')
   specs.append({'id':case,'language':lang,'split':split,'title':title,'lines':lines,'reference':'references/'+case+'.md',
    'font':('Traditional Arabic' if lang=='ar' else 'Times New Roman') if split=='validation' else ('Tahoma' if i%2 else 'Arial'),
    'background':'#ffffff' if i%2 else '#fffef9','digit_family':family,'synthetic_control':True})
blank=[]
for lang in ('ar','fr','en'):
 for split,count in (('train',4),('validation',1)):
  for i in range(count):
   case=f'blank-{lang}-{split}-{i:02d}';rng=random.Random(case)
   # Tiny random paper texture, with no erased or hidden text underneath.
   raw=bytearray()
   for _ in range(1000*720):
    v=255-rng.randrange(0,4);raw.extend((v,v,v))
   image=Image.frombytes('RGB',(1000,720),bytes(raw));p=OUT/'images'/(case+'.png');image.save(p)
   ref=OUT/'references'/(case+'.md');ref.write_text('',encoding='utf-8')
   blank.append({'id':case,'language':lang,'split':split,'document_id':case,'image':'images/'+case+'.png','image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'reference':'references/'+case+'.md','text_sha256':hashlib.sha256(b'').hexdigest(),'source':'deterministic-paper-noise-control','source_id':case,'license':'CC0-1.0','synthetic_control':True,'blank_control':True,'annotation_review':'no text was rendered'})
(OUT/'metadata/control-specs.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'metadata/blank-controls.json').write_text(json.dumps(blank,ensure_ascii=False,indent=2),encoding='utf-8')
print('Literal controls specified:',len(specs),'paper-noise blank controls rendered:',len(blank))
