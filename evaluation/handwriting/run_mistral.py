"""Run Mistral on the frozen public handwriting inputs; never send references."""
import argparse,base64,getpass,hashlib,json,os,ssl,sys,time,urllib.request,urllib.error
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--key-stdin',action='store_true');p.add_argument('--model',default='mistral-ocr-4-1');args=p.parse_args()
    manifest=json.loads((ROOT/'dataset/jeu.json').read_text(encoding='utf-8'))
    assert manifest['inspection_status']=='reviewed_before_inference'
    print('API_KEY_STDIN_READY',flush=True)
    key=(getpass.getpass('API key (hidden): ') if sys.stdin.isatty() else sys.stdin.readline()).strip() if args.key_stdin else os.environ.get('MISTRAL_API_KEY') or getpass.getpass('API key (hidden): ')
    if not key:raise RuntimeError('Missing credential')
    opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    def call(endpoint,payload=None):
        req=urllib.request.Request('https://api.mistral.ai/v1'+endpoint,data=None if payload is None else json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        started=time.perf_counter()
        try:
            with opener.open(req,timeout=240) as response:return json.load(response),{'status':response.status,'seconds':time.perf_counter()-started,'request_id':response.headers.get('x-request-id')}
        except urllib.error.HTTPError as e:raise RuntimeError('Mistral HTTP '+str(e.code)) from None
        except urllib.error.URLError as e:raise RuntimeError('Mistral network '+type(e.reason).__name__) from None
    catalog,_=call('/models');entries=[r for r in catalog.get('data',[]) if r['id']==args.model]
    assert entries,'Requested OCR model unavailable'
    write(ROOT/'results/mistral/model.json',{'requested_model':args.model,'catalog':entries[0],'observed_at_utc':datetime.now(timezone.utc).isoformat(),'immutable_weights_revision_exposed':False})
    actual_model=None
    for n,c in enumerate(manifest['cases'],1):
        dest=ROOT/'results/mistral'/(c['id']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text(encoding='utf-8'));assert old['image_sha256']==c['inference_image_sha256'];print('MISTRAL_RESUME',c['id'],flush=True);continue
        raw=(ROOT/'dataset'/c['inference_image']).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==c['inference_image_sha256']
        response,http=call('/ocr',{'model':args.model,'document':{'type':'image_url','image_url':'data:image/png;base64,'+base64.b64encode(raw).decode()},'include_image_base64':False})
        assert len(response.get('pages',[]))==1 and isinstance(response['pages'][0].get('markdown'),str)
        if actual_model:assert response['model']==actual_model
        actual_model=response['model']
        record={'id':c['id'],'model':actual_model,'image_sha256':digest,'ground_truth_given_to_model':False,'status':'ok','text':response['pages'][0]['markdown'],
                'seconds':http['seconds'],'http':http,'response':response,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        write(dest,record);dest.with_suffix('.md').write_text(record['text'],encoding='utf-8')
        write(ROOT/'results/mistral/progress.json',{'completed':n,'expected':31,'last':c['id'],'complete':n==31})
        print('MISTRAL',c['id'],str(n)+'/31',f'{http["seconds"]:.2f}s','chars='+str(len(record['text'])),flush=True)
    print('MISTRAL_ALL_31_RECORDED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:print('MISTRAL_STOPPED',type(error).__name__,str(error)[:150],flush=True);sys.exit(1)
