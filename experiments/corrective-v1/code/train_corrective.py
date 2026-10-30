"""Bounded, independent LoRA correction; never modifies the historical run.

Publisher transcriptions are supervision only. Generation receives image+prompt.
No new packages, no S3 secrets, no automatic promotion, no silent truncation.
"""
import hashlib,json,math,os,random,subprocess,time
from collections import defaultdict
from pathlib import Path

DATA=Path('/export/ocr-fidelity-corrective-v2')
RUN=Path('/export/ocr-fidelity-corrective-run-v1')
BASE=Path('/export/markdown-best-001442/base-reference')
AUDIT=Path('/export/ocr-fidelity-20261027-v1')
CODE=Path('/corrective-code')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def log(event,**data):
    record={'event':event,'timestamp':time.time(),**data}
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print('CORRECTIVE',json.dumps(record,ensure_ascii=False),flush=True)

assert not RUN.exists(),'Do not overwrite or implicitly restart an existing correction'
proof=json.loads((DATA/'corpus-ready.json').read_text())
assert proof['state']=='ready_for_bounded_corrective_training' and proof['all_supervised_end_tokens_verified']
assert proof['corpus_manifest_sha256']==sha(DATA/'manifest.json')
devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
assert len(devices)==1 and 'H200' in devices[0] and int(devices[0].split(',')[-1])<512
assert 'GPU-557da63b-03d0-6a77-e91b-de84f7d7a83b' not in devices[0]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Allocated GPU has an existing compute process'
RUN.mkdir();(RUN/'checkpoints').mkdir()
for cache in ('/tmp/corrective-xdg/torch/kernels','/tmp/corrective-triton'):Path(cache).mkdir(parents=True,exist_ok=True)
for item in json.loads(Path('/export/benchmark-31-20261026-v1/base-files.json').read_text()):assert sha(BASE/item['name'])==item['sha256']

import torch,triton,transformers
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,StoppingCriteria,StoppingCriteriaList
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from lora import LowRankLinear
from ocr_fidelity import TRAINING_MARKDOWN_PROMPT,literal_prompt,inspect_output,repetition_flags,arabic_letters,is_exact_white_blank

assert triton.__version__=='3.7.1' and qwen.is_fast_path_available
torch.set_num_threads(4);random.seed(20261027);torch.manual_seed(20261027)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
model,info=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
model.to('cuda');model.requires_grad_(False);model.eval()
manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
cached={}
def clear_rope():
    for m in model.modules():
        if hasattr(m,'rope_deltas'):m.rope_deltas=None
def encode(c):
    if c['id'] in cached:return cached[c['id']]
    assert sha(DATA/c['image'])==c['image_sha256'] and sha(DATA/c['reference'])==c['text_sha256']
    with Image.open(DATA/c['image']) as im:image=im.convert('RGB')
    text=(DATA/c['reference']).read_text(encoding='utf-8')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':literal_prompt(TRAINING_MARKDOWN_PROMPT,c['prompt_language'])}]}]
    a=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False,add_generation_prompt=True)
    b=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':text}]}],tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False,add_generation_prompt=False)
    n=a['input_ids'].shape[1];assert torch.equal(a['input_ids'],b['input_ids'][:,:n]) and b['input_ids'].shape[1]<=8192
    b['labels']=b['input_ids'].clone();b['labels'][:,:n]=-100
    assert int((b['labels']==processor.tokenizer.convert_tokens_to_ids('<|im_end|>')).sum())==1
    cached[c['id']]=dict(b);return cached[c['id']]
def batch(c):return {k:v.to('cuda') for k,v in encode(c).items()}

short=next(c for c in cases if c['id'].startswith('control-fr-train'))
clear_rope()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):native_loss=float(model(**batch(short),use_cache=False).loss)
count=0
for name,module in list(model.named_modules()):
    if isinstance(module,torch.nn.Linear) and 'language_model.layers.' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}:
        parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,LowRankLinear(module,rank=16,alpha=32,compute_dtype='fp32'));count+=1
assert count==128
params=[p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))]
assert len(params)==256 and sum(p.numel() for p in params)==29097984 and all(p.dtype==torch.float32 for p in params)
clear_rope()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):wrapped_loss=float(model(**batch(short),use_cache=False).loss)
assert abs(native_loss-wrapped_loss)<1e-5 and math.isfinite(wrapped_loss)
log('base_identity_preflight',native_loss=native_loss,wrapped_loss=wrapped_loss,lora_modules=count)
model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
config={'phase':'bounded-public-literal-correction','initialization':'verified original Qwen3.5-9B + fresh zero-B LoRA; regressed adapter NOT inherited',
 'historical_checkpoint_preserved':4796,'epochs':2,'train_examples_per_epoch':300,'validation_examples':75,'global_batch':16,'accumulation':16,'microbatch':1,'world_size':1,
 'maximum_sequence_tokens':8192,'base_dtype':'bf16','lora_compute_dtype':'fp32','lora_rank':16,'lora_alpha':32,'max_lr':1e-5,'min_lr':2e-6,'warmup_updates':4,'clip':1.,
 'seed':20261027,'corpus_manifest_sha256':sha(DATA/'manifest.json'),'code_sha256':{p.name:sha(p) for p in (CODE/'train_corrective.py',CODE/'ocr_fidelity.py',CODE/'lora.py')},
 'gpu':devices[0],'torch':torch.__version__,'transformers':transformers.__version__,'deployment':'candidate only; no automatic promotion',
 'licenses':manifest['source_rights'],'publisher_annotations_not_all_independently_retranscribed':True}
write(RUN/'config.json',config)
step=0;seen=0;max_steps=38;best=None;history=[]
def save(label):
    path=RUN/'checkpoints'/('adapter-'+label+'.safetensors');assert not path.exists()
    adapter={'hf.'+n:p.detach().cpu().contiguous().clone() for n,p in model.named_parameters() if p.requires_grad}
    assert len(adapter)==256;save_file(adapter,str(path))
    metadata={'label':label,'checkpoint':'corrective-public-v1/'+label,'phase_step':step,'phase_config':config,'step':step,'examples_consumed':seen,'adapter_path':str(path),'adapter_sha256':sha(path),'config_sha256':sha(RUN/'config.json'),'base_manifest_sha256':sha('/export/benchmark-31-20261026-v1/base-files.json')}
    write(path.with_suffix('.json'),metadata);return metadata
def evaluate(label):
    model.eval();stats=defaultdict(lambda:[0.,0,0])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in cases:
            if c['split']!='validation':continue
            clear_rope();b=batch(c);tokens=int((b['labels'][:,1:]!=-100).sum());loss=float(model(**b,use_cache=False).loss)
            assert math.isfinite(loss)
            for key in (c['language'],'all'):
                stats[key][0]+=loss*tokens;stats[key][1]+=tokens;stats[key][2]+=1
            del b
    values={k:{'loss':v[0]/v[1],'supervised_tokens':v[1],'examples':v[2]} for k,v in stats.items()}
    result={'label':label,'step':step,'examples_consumed':seen,'values':values};history.append(result);write(RUN/'eval-history.json',history);log('validation',**result)
    return result

inputs=json.loads((CODE/'inputs.json').read_text());assert inputs['references_not_in_model_inputs']
eos=list(dict.fromkeys(x for x in (processor.tokenizer.eos_token_id,processor.tokenizer.convert_tokens_to_ids('<|im_end|>')) if isinstance(x,int) and x>=0))
def generate_checks(label,ids):
    model.eval();model.gradient_checkpointing_disable()
    for c in inputs['cases']:
        if c['id'] not in ids:continue
        assert sha(c['image'])==c['image_sha256']
        with Image.open(c['image']) as im:image=im.convert('RGB')
        target=RUN/'generations'/label/(c['id']+'.json');assert not target.exists()
        if is_exact_white_blank(image):
            write(target,{'id':c['id'],'text':'','stop_reason':'verified_exact_blank','model_invoked':False,'image_sha256':c['image_sha256'],'ground_truth_given_to_model':False});continue
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':literal_prompt(TRAINING_MARKDOWN_PROMPT,c['language'])}]}]
        encoded=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False,add_generation_prompt=True)
        n=encoded['input_ids'].shape[1];assert n+2048<=8192
        class Guard(StoppingCriteria):
            reason=None
            def __call__(self,ids,scores,**kw):
                if (ids.shape[1]-n)%8:return False
                text=processor.tokenizer.decode(ids[0,n:],skip_special_tokens=True)
                if repetition_flags(text):self.reason='repetition'
                elif c['language'] in ('fr','en') and arabic_letters(text)>=4:self.reason='unexpected_arabic_text'
                return self.reason is not None
        guard=Guard();clear_rope();started=time.perf_counter();log('generation_start',label=label,id=c['id'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(**{k:v.to('cuda') for k,v in encoded.items()},max_new_tokens=2048,max_time=180,do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.,no_repeat_ngram_size=0,eos_token_id=eos,pad_token_id=eos[0],stopping_criteria=StoppingCriteriaList([guard]))
        tokens=output[0,n:].cpu().tolist();text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
        reason='eos' if tokens and tokens[-1] in eos else guard.reason or ('token_limit' if len(tokens)>=2048 else 'time_limit_or_other')
        result={'id':c['id'],'text':text,'stop_reason':reason,'tokens':len(tokens),'seconds':time.perf_counter()-started,'image_sha256':c['image_sha256'],'fidelity_audit':inspect_output(text,c['language'],reason),'ground_truth_given_to_model':False}
        write(target,result);target.with_suffix('.md').write_text(text,encoding='utf-8');log('generation_done',label=label,id=c['id'],stop_reason=reason,seconds=result['seconds'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

baseline=evaluate('base-initial');save('initial-zero-B')
for epoch in range(2):
    groups={lang:[c for c in cases if c['split']=='train' and c['language']==lang] for lang in ('ar','fr','en')}
    rng=random.Random(20261027+epoch)
    for group in groups.values():rng.shuffle(group)
    ordered=[groups[lang][i] for i in range(100) for lang in ('ar','fr','en')]
    model.train();optimizer.zero_grad(set_to_none=True)
    for offset in range(0,len(ordered),16):
        current=ordered[offset:offset+16];started=time.perf_counter();loss_sum=0.;tokens_sum=0
        rate=1e-5*(step+1)/4 if step<4 else 2e-6+(1e-5-2e-6)*.5*(1+math.cos(math.pi*(step-4)/(max_steps-4)))
        for group in optimizer.param_groups:group['lr']=rate
        for c in current:
            clear_rope();b=batch(c);tokens_sum+=int((b['labels'][:,1:]!=-100).sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**b,use_cache=False).loss
            assert bool(torch.isfinite(loss));loss_sum+=float(loss.detach());(loss/len(current)).backward();del loss,b
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert math.isfinite(float(norm)) and float(norm)>0
        if step==0:
            assert any(p.grad is not None and bool((p.grad!=0).any()) for p in params),'No real gradient'
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert all(bool(torch.isfinite(p).all()) for p in params)
        step+=1;seen+=len(current)
        log('update',epoch=epoch+1,step=step,examples_consumed=seen,examples=len(current),train_loss_mean=loss_sum/len(current),supervised_tokens=tokens_sum,lr=rate,gradient_norm=float(norm),seconds=time.perf_counter()-started)
        write(RUN/'progress.json',{'state':'training','epoch':epoch+1,'step':step,'examples_consumed':seen,'total_examples':600,'last_train_loss':loss_sum/len(current)})
    marker=save('epoch-'+str(epoch+1));evaluation=evaluate('epoch-'+str(epoch+1))
    if best is None or evaluation['values']['all']['loss']<best['evaluation']['values']['all']['loss']:best={'checkpoint':marker,'evaluation':evaluation}
    generate_checks('epoch-'+str(epoch+1),{'ar-02','fr-01','en-06','probe-fr-symboles','probe-chiffres','probe-vide'} if epoch==0 else {c['id'] for c in inputs['cases']})
assert step==38 and seen==600
torch.save({'optimizer':optimizer.state_dict(),'step':step,'examples_consumed':seen,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':config},RUN/'optimizer-final.pt')
write(RUN/'training-complete.json',{'state':'bounded_training_complete_pending_generation_review','updates':step,'examples_consumed':seen,'unique_training_pages':300,'epochs':2,'validation_pages':75,'best_by_loss':best,'baseline':baseline,'historical_checkpoint_unchanged':True,'automatically_promoted':False})
log('bounded_training_complete',updates=step,examples_consumed=seen,promotion='pending generation and glyph review')
