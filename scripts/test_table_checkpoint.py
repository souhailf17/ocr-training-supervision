"""Bounded, single-free-GPU diagnostic. Image-only inputs; no training or GT access."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from annotation_targets import TABLE_OCR_PROMPT


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=8192)
    parser.add_argument('--max-page-seconds',type=int,default=300)
    parser.add_argument('--s3-prefix',default='')
    args=parser.parse_args()
    if not 1<=args.max_new_tokens<=8192 or not 1<=args.max_page_seconds<=300:
        raise ValueError('Diagnostic bounds: 8192 output tokens, 300 seconds/page maximum')
    if args.output_dir.exists():raise ValueError('Output run already exists')
    manifest=json.loads(args.inputs.read_text(encoding='utf-8'))
    if manifest.get('ground_truth_in_model_inputs') is not False or manifest['prompt']!=TABLE_OCR_PROMPT:
        raise ValueError('Expected the image-only table diagnostic manifest')
    rows=manifest['records']
    if not 1<=len(rows)<=4:raise ValueError('Diagnostic is limited to four images')
    allowed={'id','image_path','image_sha256','width','height','reference_available'}
    if any(set(r)-allowed for r in rows):raise ValueError('Model inputs must not contain annotations/ground truth')
    if any(not isinstance(r.get('id'),str) or not re.fullmatch(r'sample_\d+',r['id']) for r in rows):
        raise ValueError('Expected safe supplied-sample identifiers')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate diagnostic image IDs')
    # Check before importing a model/initializing CUDA. Never borrow a busy device.
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    visible=[line.split(',') for line in gpu.splitlines() if line.strip()]
    if len(visible)!=1 or int(visible[0][1].strip())>256:
        raise RuntimeError('The pod must expose exactly one currently unused GPU')
    processes=subprocess.check_output(['nvidia-smi','--id='+visible[0][0].strip(),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if processes.strip():raise RuntimeError('Assigned GPU already has a compute process; refusing to use it')
    if digest(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Checkpoint SHA mismatch')
    for row in rows:
        path=(args.inputs.parent/row['image_path']).resolve()
        if not path.is_relative_to(args.inputs.parent.resolve()) or digest(path)!=row['image_sha256']:
            raise ValueError('Unsafe or changed input image')
    import math
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    if torch.cuda.device_count()!=1:raise RuntimeError('One assigned GPU required')
    torch.manual_seed(42);device=torch.device('cuda',0)

    class LowRankLinear(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.scale=2.
            self.lora_a=torch.nn.Parameter(torch.empty(16,base.in_features,device=base.weight.device,dtype=torch.float32))
            self.lora_b=torch.nn.Parameter(torch.zeros(base.out_features,16,device=base.weight.device,dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
        @property
        def weight(self):return self.base.weight
        @property
        def bias(self):return self.base.bias
        def forward(self,x):
            result=self.base(x)
            with torch.autocast('cuda',enabled=False):delta=F.linear(F.linear(x.float(),self.lora_a),self.lora_b)*self.scale
            return result+delta.to(result.dtype)

    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    adapters=state['adapter']
    if len(adapters)!=256 or not all(torch.isfinite(p).all() for p in adapters.values()):
        raise ValueError('Expected the verified 128-module custom FP32 LoRA checkpoint')
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    count=0
    for name,module in list(model.named_modules()):
        if isinstance(module,torch.nn.Linear) and 'language_model.layers.' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}:
            parent,attribute=name.rsplit('.',1);setattr(model.get_submodule(parent),attribute,LowRankLinear(module));count+=1
    if count!=128:raise ValueError('Adapter module layout changed')
    used=set()
    for name,parameter in model.named_parameters():
        if name.endswith(('lora_a','lora_b')):
            key=next((p+name for p in ('module.hf.','hf.') if p+name in adapters),None)
            if key is None or adapters[key].shape!=parameter.shape:raise ValueError('Adapter key/shape mismatch')
            with torch.no_grad():parameter.copy_(adapters[key])
            if not torch.equal(parameter.detach().cpu(),adapters[key]):raise ValueError('Adapter copy differs')
            used.add(key)
    if used!=set(adapters):raise ValueError('An adapter tensor was not loaded')
    step=state['clock']['step_index']
    corrected_samples=state.get('phase_samples',0) if state.get('phase_config',{}).get('phase')=='tables-only-correction' else 0
    correction_data_sha=state.get('data_complete_sha256') if corrected_samples else None
    del state,adapters
    model.requires_grad_(False);model.eval()
    eos=[processor.tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|endoftext|>')]
    if any(t is None or t==processor.tokenizer.unk_token_id for t in eos):raise ValueError('EOS token mismatch')
    args.output_dir.mkdir(parents=True)
    s3=None
    if args.s3_prefix:
        import boto3
        from botocore.config import Config
        s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',config=Config(connect_timeout=10,read_timeout=30,retries={'max_attempts':3},s3={'addressing_style':'path'}))
    def save(name,body):
        path=args.output_dir/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body,encoding='utf-8')
        if s3 is not None:
            key=args.s3_prefix.rstrip('/')+'/'+name
            try:s3.head_object(Bucket='qwen35-9b-ocr-finetuning',Key=key)
            except s3.exceptions.ClientError as exc:
                if exc.response['Error']['Code'] not in ('404','NoSuchKey'):raise
            else:raise ValueError('Refusing to overwrite an existing diagnostic S3 object')
            sha=digest(path)
            s3.upload_file(str(path),'qwen35-9b-ocr-finetuning',key,ExtraArgs={'Metadata':{'sha256':sha}})
            response=s3.get_object(Bucket='qwen35-9b-ocr-finetuning',Key=key)
            try:uploaded=response['Body'].read()
            finally:response['Body'].close()
            if hashlib.sha256(uploaded).hexdigest()!=sha:raise ValueError('Prediction S3 verification failed')
    config={'checkpoint_sha256':args.checkpoint_sha256,'global_step':step,'prompt':TABLE_OCR_PROMPT,
            'images':len(rows),'max_new_tokens':args.max_new_tokens,'max_page_seconds':args.max_page_seconds,
            'ground_truth_in_prompt':False,'training':False,'resolution':'Supplied original PNG; saved processor resize only',
            'stopping_tokens':eos,'corrected_labels_trained':corrected_samples>0,
            'corrective_training_samples':corrected_samples,'corrective_data_sha256':correction_data_sha,
            'purpose':'Diagnostic of the selected checkpoint; corrected supervision metadata does not prove table completeness or independent held-out quality.'}
    save('config.json',json.dumps(config,ensure_ascii=False,indent=2))
    results=[]
    for row in rows:
        with Image.open(args.inputs.parent/row['image_path']) as image:im=image.convert('RGB')
        if im.size!=(row['width'],row['height']):raise ValueError('Input dimensions changed')
        messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':TABLE_OCR_PROMPT}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False,add_generation_prompt=True).to(device)
        if inputs['input_ids'].shape[1]+args.max_new_tokens>=model.config.text_config.max_position_embeddings:
            raise ValueError('Generation would exceed model context')
        torch.cuda.synchronize();started=time.time()
        with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,max_time=args.max_page_seconds,do_sample=False,use_cache=True,eos_token_id=eos)
        torch.cuda.synchronize();seconds=time.time()-started
        ids=generated[0,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,t in enumerate(ids) if t in eos),None)
        if end is not None:ids=ids[:end+1]
        prediction=processor.decode(ids,skip_special_tokens=True)
        save('predictions/'+row['id']+'.md',prediction)
        record={'id':row['id'],'seconds':seconds,'tokens':len(ids),'stop_reason':'eos' if end is not None else 'token_limit' if len(ids)>=args.max_new_tokens else 'time_limit','prediction_chars':len(prediction)}
        results.append(record);print('TABLE_DIAGNOSTIC_PAGE',json.dumps(record),flush=True)
    save('result.json',json.dumps({'state':'complete','config':config,'records':results},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
