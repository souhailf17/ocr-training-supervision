"""Image-only, offline, resumable Unlimited-OCR-2 benchmark on one GPU.

Uses reviewed pinned official source and native single-page infer(). References
are not part of inputs.json and are never read by this inference process.
No installation, download, server creation, training, or other workload changes.
"""
import argparse,contextlib,gc,hashlib,json,os,signal,subprocess,time
from pathlib import Path
REVISION='5632329a32e103d5f1add270681e0f5c1233c6e4'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):raise TimeoutError('Inference wall-clock budget exceeded')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--model-dir',type=Path);p.add_argument('--model-lock',type=Path)
    p.add_argument('--inputs',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--ids',nargs='*')
    a=p.parse_args()
    if a.check:
        from verifier_unlimited import main as check
        check();return
    for k in ('model_dir','model_lock','inputs','output_dir'):
        if getattr(a,k) is None:p.error('--'+k.replace('_','-')+' is required')
    if a.output_dir.exists() and not a.resume:raise FileExistsError('Use a new output or explicitly --resume; do not overwrite previous results')
    lock=json.loads(a.model_lock.read_text(encoding='utf-8'));assert lock['repo']=='baidu/Unlimited-OCR-2' and lock['revision']==REVISION
    for f in lock['files']:assert sha(a.model_dir/f['name'])==f['sha256'],'Pinned model/source file digest differs'
    source=json.loads(a.inputs.read_text(encoding='utf-8'));assert source['references_not_in_model_inputs'] and len(source['cases'])==31
    assert all(set(c)<= {'id','language','image','image_sha256'} for c in source['cases']),'Generation manifest must not contain references'
    rows=[c for c in source['cases'] if not a.ids or c['id'] in a.ids];assert rows
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(devices)==1 and int(devices[0].split(',')[-1])<512,'Allocate exactly one unused GPU'
    assert 'GPU-557da63b-03d0-6a77-e91b-de84f7d7a83b' not in devices[0],'Historical training GPU excluded'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Allocated GPU has an existing compute process'
    import torch,transformers
    from PIL import Image
    from transformers import AutoTokenizer,AutoModel
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output_dir.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=True)
    model,loading=AutoModel.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=True,use_safetensors=True,torch_dtype=torch.bfloat16,output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Model checkpoint did not load exactly'
    model.eval().cuda();model.requires_grad_(False)
    protocol={'model':'baidu/Unlimited-OCR-2','revision':REVISION,'model_lock_sha256':sha(a.model_lock),'inputs_sha256':sha(a.inputs),'torch':torch.__version__,'transformers':transformers.__version__,
      'gpu':devices[0].rsplit(',',1)[0].strip(),'precision':'bf16','mode':'native infer; automatic single-page gundam tiling','prompt':'native default Multi page parsing.','max_new_tokens':6000,'context_limit':8192,'max_seconds':300,
      'temperature':0.,'no_repeat_ngram_size':0,'page_count_guard':True,'max_page_tokens':6000,'ground_truth_given_to_model':False,'training_modified':False}
    audit=a.output_dir/'model.json'
    if audit.exists():assert json.loads(audit.read_text())==protocol,'Cannot resume a changed protocol'
    else:write(audit,protocol)
    for c in rows:
        image=Path(c['image']);image=image if image.is_absolute() else a.inputs.parent/image
        assert sha(image)==c['image_sha256']
        with Image.open(image) as im:assert im.mode=='RGB' and max(im.size)<=1024
        target=a.output_dir/(c['id']+'.json')
        if target.exists():
            old=json.loads(target.read_text(encoding='utf-8'));assert old['image_sha256']==c['image_sha256']
            print('ALREADY_RECORDED',c['id'],old['status'],flush=True);continue
        artifact=a.output_dir/'native-artifacts'/c['id'];assert not artifact.exists(),'Uncertain interrupted image: inspect it before creating a separate retry'
        artifact.mkdir(parents=True);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        print('UNLIMITED_START',c['id'],flush=True)
        result={'id':c['id'],'image_sha256':c['image_sha256'],'ground_truth_given_to_model':False,'model_revision':REVISION,'complete_ocr_proven':False}
        try:
            with deadline(300),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                response=model.infer(tokenizer,image_file=str(image),output_path=str(artifact),save_results=True,eval_mode=True,return_dict=True,
                  max_length=8192,max_new_tokens=6000,temperature=0.,no_repeat_ngram_size=0,max_page_tokens=6000,page_count_guard=True)
            assert isinstance(response['text'],str) and len(response['token_ids'])<=6000
            cleaned=(artifact/'result.md').read_text(encoding='utf-8').replace('<PAGE>','').strip()
            result.update(text=cleaned,raw_model_text=response['text'],status='ok',stop_reason=response['finish_reason'],generated_tokens=len(response['token_ids']),
              prompt_tokens=len(response['prompt_token_ids']),seconds=time.perf_counter()-started,vram_gib=torch.cuda.max_memory_allocated()/2**30,
              normalization='official saved Markdown (grounding tags removed and native equivalent math operator spellings applied); nonprinted <PAGE> separators removed; raw retained')
            assert result['prompt_tokens']+result['generated_tokens']<=8192
            write(artifact/'native-response.json',response);target.with_suffix('.md').write_text(cleaned,encoding='utf-8');target.with_suffix('.raw.md').write_text(response['text'],encoding='utf-8')
        except (TimeoutError,torch.OutOfMemoryError,RuntimeError) as error:
            result.update(text=None,status='error',error_type=type(error).__name__,seconds=time.perf_counter()-started)
            write(target,result);print('UNLIMITED_STOP',c['id'],result['error_type'],flush=True);raise
        write(target,result);print('UNLIMITED_DONE',c['id'],result['stop_reason'],result['generated_tokens'],flush=True)
        gc.collect();torch.cuda.empty_cache()
    write(a.output_dir/'completion.json',{'attempted':len(list(a.output_dir.glob('[a-z][a-z]-[0-9][0-9].json'))),'expected':31,'all_images_not_guaranteed_complete':True})
if __name__=='__main__':main()
