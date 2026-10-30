"""Local final-adapter OCR with literal text preservation and explicit review state.

No installation, automatic download, training, post-translation or digit repair.
"""
import argparse,hashlib,importlib.metadata,json,time
from pathlib import Path
from ocr_fidelity import TRAINING_MARKDOWN_PROMPT,literal_prompt,StreamingGuard,inspect_output,is_exact_white_blank

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--check',action='store_true');p.add_argument('--model-dir',type=Path);p.add_argument('--adapter',type=Path);p.add_argument('--snapshot',type=Path)
 p.add_argument('--base-manifest',type=Path);p.add_argument('--input',type=Path);p.add_argument('--page',type=int,default=1);p.add_argument('--language',choices=['ar','fr','en','mixed'],default='mixed')
 p.add_argument('--output',type=Path);p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--max-seconds',type=float,default=180)
 a=p.parse_args()
 if a.check:
  for name in ('torch','transformers','safetensors','Pillow','PyMuPDF'):
   try:print(name,importlib.metadata.version(name))
   except importlib.metadata.PackageNotFoundError:print(name,'ABSENT')
  print('No installation, network request or model loading.');return
 for name in ('model_dir','adapter','snapshot','input','output'):
  if getattr(a,name) is None:p.error('--'+name.replace('_','-')+' is required')
 if not 1<=a.max_new_tokens<=8192 or not 0<a.max_seconds<=1800:p.error('Invalid generation limits')
 if a.page<1:p.error('Page numbers start at one')
 report_path=a.output.with_suffix('.json');raw_path=a.output.with_suffix('.raw.md')
 if report_path.exists() or raw_path.exists():raise FileExistsError('Choose a new output; recorded OCR is never overwritten')
 from PIL import Image
 if a.input.suffix.lower()=='.pdf':
  import pymupdf
  with pymupdf.open(a.input) as pdf:
   if a.page>len(pdf):raise ValueError('PDF page does not exist')
   pix=pdf[a.page-1].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False)
   image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
 else:
  with Image.open(a.input) as src:image=src.convert('RGB')
 # Exact blank detection precedes resizing, so downscaling cannot erase faint ink.
 if is_exact_white_blank(image):
  raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_text('',encoding='utf-8')
  report_path.write_text(json.dumps({'status':'verified_exact_blank','raw_text':'','source':'exact_input_pixels_no_model','input_sha256':sha(a.input),'page':a.page,'model_loaded':False,'text_rewritten':False},indent=2),encoding='utf-8')
  print('Exact white blank: empty output; no model loaded.');return
 image.thumbnail((1024,1024))
 import torch
 from safetensors.torch import load_file
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,StoppingCriteria,StoppingCriteriaList
 from lora import LowRankLinear
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16 CUDA GPU is required')
 snapshot=json.loads(a.snapshot.read_text(encoding='utf-8'));assert sha(a.adapter)==snapshot['adapter_sha256']
 assert sha(Path(__file__).with_name('lora.py'))==snapshot['phase_config']['code_sha256']['lora.py']
 base_manifest=a.base_manifest or a.snapshot.with_name('base-files.json')
 if not base_manifest.is_file():raise ValueError('Supply the verified base-files.json manifest with --base-manifest')
 for item in json.loads(base_manifest.read_text(encoding='utf-8')):assert sha(a.model_dir/item['name'])==item['sha256'],'Base model differs from the verified training base'
 processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
 prompt=literal_prompt(TRAINING_MARKDOWN_PROMPT,a.language)
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
 encoded=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False,add_generation_prompt=True)
 n=int(encoded['input_ids'].shape[1])
 if n+a.max_new_tokens>8192:raise ValueError('Input + output exceed the unchanged 8192-token context; no silent truncation')
 model,info=Qwen3_5ForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
 assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
 model.to('cuda');model.requires_grad_(False);adapters=load_file(str(a.adapter),device='cpu');count=0
 for name,module in list(model.named_modules()):
  if isinstance(module,torch.nn.Linear) and 'language_model.layers.' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}:
   parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,LowRankLinear(module,rank=16,alpha=32,compute_dtype='fp32'));count+=1
 assert count==128;used=set()
 with torch.no_grad():
  for name,param in model.named_parameters():
   if name.endswith(('.lora_a','.lora_b')):
    key='hf.'+name;param.copy_(adapters[key]);assert torch.equal(param.detach().cpu(),adapters[key]);used.add(key)
 assert len(used)==256 and used==set(adapters);del adapters;model.requires_grad_(False);model.eval()
 guard=StreamingGuard(a.language)
 class Criterion(StoppingCriteria):
  def __call__(self,ids,scores,**kwargs):
   return (ids.shape[1]-n)%8==0 and guard.check(processor.tokenizer.decode(ids[0,n:],skip_special_tokens=True))
 eos=list(dict.fromkeys(x for x in (processor.tokenizer.eos_token_id,processor.tokenizer.convert_tokens_to_ids('<|im_end|>')) if isinstance(x,int) and x>=0 and x!=processor.tokenizer.unk_token_id))
 torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  output=model.generate(**{k:v.to('cuda') for k,v in encoded.items()},max_new_tokens=a.max_new_tokens,max_time=a.max_seconds,do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.0,no_repeat_ngram_size=0,eos_token_id=eos,pad_token_id=eos[0],stopping_criteria=StoppingCriteriaList([Criterion()]))
 ids=output[0,n:].cpu().tolist();text=processor.tokenizer.decode(ids,skip_special_tokens=True)
 reason='eos' if ids and ids[-1] in eos else guard.reason or ('token_limit' if len(ids)>=a.max_new_tokens else 'time_limit_or_other')
 audit=inspect_output(text,a.language,reason)
 data={'status':'requires_review' if audit['requires_review'] else 'generated_unverified','checkpoint':snapshot['checkpoint'],'input_sha256':sha(a.input),'page':a.page,'language_hint':a.language,'stop_reason':reason,'fidelity_audit':audit,'seconds':time.perf_counter()-start,'raw_text':text,'image_grounding_verified':False}
 raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_text(text,encoding='utf-8');report_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'status':data['status'],'stop_reason':reason,'flags':audit['flags'],'raw_output':str(raw_path)},ensure_ascii=False))
 if audit['requires_review']:raise SystemExit(2)

if __name__=='__main__':main()
