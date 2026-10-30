"""Sequential inference requests to the existing service, without GPU allocation."""
import base64,hashlib,json,sys,time,urllib.error
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,'/home/admin_mtnra/unlimited-ocr2-serve-20261028')
from ingress_http import http
ROOT=Path(__file__).resolve().parent
BASE='https://unlimited-ocr2-mtrna-ns1.apps.olympus.atlasxai.ma'
def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
def main():
    manifest=json.loads((ROOT/'inference-manifest.json').read_text())
    assert manifest['references_included'] is False and len(manifest['cases'])==31
    results=ROOT/'results';results.mkdir(exist_ok=True)
    health=http(BASE,'/ready');assert health['ready'] and health['model']=='unlimited-ocr2'
    write(results/'model.json',{'base':BASE,'model':'unlimited-ocr2','health_before':health,'source':'existing deployed service','new_gpu_allocated':False,'training_or_deployment_modified':False,'requested_max_tokens':6000,'ocr_mode':'raw with native decoded text evaluated','started_at_utc':datetime.now(timezone.utc).isoformat()})
    for n,c in enumerate(manifest['cases'],1):
        dest=results/(c['id']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text());assert old['image_sha256']==c['image_sha256'];print('UNLIMITED_RESUME',c['id'],flush=True);continue
        raw=(ROOT/c['image']).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==c['image_sha256']
        payload={'model':'unlimited-ocr2','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}},{'type':'text','text':'Extract the text content from this image.'}]}],'temperature':0,'max_tokens':6000,'ocr_mode':'raw'}
        started=time.perf_counter()
        print('UNLIMITED_REQUEST',c['id'],str(n)+'/31',flush=True)
        try:
            response=http(BASE,'/v1/chat/completions',payload,timeout=920)
            assert response['ocr']['model_revision']=='5632329a32e103d5f1add270681e0f5c1233c6e4'
            reason=response['choices'][0]['finish_reason']
            record={'id':c['id'],'status':'ok','model':'baidu/Unlimited-OCR-2','image_sha256':digest,'ground_truth_given_to_model':False,
                    'text':response['ocr']['text'],'raw_model_text':response['ocr']['raw'],'stop_reason':reason,'normal_stop':reason=='stop',
                    'seconds':time.perf_counter()-started,'response':response,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            record={'id':c['id'],'status':'transport_or_service_error','image_sha256':digest,'ground_truth_given_to_model':False,'text':None,
                    'error_type':type(e).__name__,'seconds':time.perf_counter()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        write(dest,record)
        if isinstance(record.get('text'),str):dest.with_suffix('.md').write_text(record['text'])
        write(results/'progress.json',{'attempted':n,'expected':31,'last':c['id'],'complete':n==31,'last_status':record['status']})
        print('UNLIMITED_RECORDED',c['id'],record['status'],round(record['seconds'],2),'seconds','chars',len(record['text']) if isinstance(record['text'],str) else None,flush=True)
    print('UNLIMITED_ALL_31_ATTEMPTED',flush=True)
if __name__=='__main__':main()
