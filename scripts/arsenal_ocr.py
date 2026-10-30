"""Real Arsenal trainer: BF16 Qwen OCR with float32 low-rank adapters."""
import hashlib, io, json, os, random
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from lora import LowRankLinear
from arsenal.config.base import BaseConfig
from arsenal.config.compile import CompileConfig
from arsenal.config.dataloader import BaseDataloaderConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.optimizer import OptimizerConfig
from arsenal.data import BaseDataloader, BaseBatch
from arsenal.model import ArsenalModule
from arsenal.trainer import BaseTrainer
from lr_scheduler_basic import CosineSchedulerConfig
from stage_training import ROOT, BUCKET
from s3_common import client
from annotation_targets import assert_no_table_placeholders, TARGET_FORMAT, TABLE_OCR_PROMPT

class OCRModel(ArsenalModule):
    model_path: str
    def setup(self):
        # Materialization loads the pinned pretrained model exactly once.
        self.hf = torch.nn.Identity()
    def materialize(self, device, dtype):
        with torch.device('cpu'):
            self.hf = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_path, local_files_only=True, dtype=dtype, attn_implementation='sdpa')
        self.hf.to(device)
        self.hf.requires_grad_(False)
        lora_compute_dtype = os.environ.get('Q35OCR_LORA_COMPUTE_DTYPE', 'fp32').lower()
        if lora_compute_dtype not in {'fp32', 'bf16'}:
            raise ValueError('Q35OCR_LORA_COMPUTE_DTYPE must be fp32 or bf16')
        targets=[]
        for name, module in list(self.hf.named_modules()):
            if isinstance(module, torch.nn.Linear) and 'language_model.layers.' in name:
                # Only projections called through forward; fused delta-net inputs
                # are excluded because some kernels access weights directly.
                if name.rsplit('.',1)[-1] not in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}: continue
                parent, attr = name.rsplit('.',1)
                setattr(self.hf.get_submodule(parent),attr,LowRankLinear(module,compute_dtype=lora_compute_dtype))
                targets.append(name)
        assert targets, 'No verified language projection targets'
        self.hf.config.use_cache=False
        self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.hf.enable_input_require_grads()
        (ROOT/'logs/adapter-targets.json').write_text(json.dumps(targets,indent=2))
        print('LORA_COMPUTE_DTYPE',lora_compute_dtype,flush=True)
        print('ADAPTER_TARGETS',len(targets),'TRAINABLE',sum(p.numel() for p in self.parameters() if p.requires_grad),flush=True)
    def __call__(self, batch): return self.hf(**batch, use_cache=False).loss

class OCRBatch(dict):
    def to(self, device, non_blocking=False):
        return OCRBatch({k:v.to(device=device,non_blocking=non_blocking) for k,v in self.items()})
    def pin_memory(self):
        return OCRBatch({k:(v.pin_memory() if isinstance(v,torch.Tensor) and v.device.type=='cpu' else v) for k,v in self.items()})

def collate_ocr(rows, pad_token_id, pin_memory=True):
    """Pad and concatenate examples, then pin the final batch once."""
    if not rows:
        raise ValueError('Cannot collate an empty OCR batch')
    result={}; width=max(row['input_ids'].shape[1] for row in rows)
    for key in rows[0]:
        values=[row[key] for row in rows]
        if key in ('input_ids','attention_mask','labels','mm_token_type_ids','token_type_ids'):
            fill=-100 if key=='labels' else pad_token_id if key=='input_ids' else 0
            values=[torch.nn.functional.pad(value,(0,width-value.shape[1]),value=fill) for value in values]
        elif key not in ('pixel_values','image_grid_thw'):
            raise ValueError('Unvalidated processor batch field: '+key)
        result[key]=torch.cat(values,dim=0)
    batch=OCRBatch(result)
    return batch.pin_memory() if pin_memory else batch

class OCRLoader(BaseDataloader):
    def __init__(self, path, model_path, split='train'):
        self.rows=[json.loads(x) for x in Path(path).read_text().splitlines() if json.loads(x)['split']==split]
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        self.cursor=0
    def state_dict(self): return {'cursor':self.cursor}
    def load_state_dict(self,state): self.cursor=state['cursor']
    def encode(self, row, raw=None):
        # Fail before a forward/update if an old target still teaches file links
        # in place of the actual cells. Historical code remains on the old branch.
        assert_no_table_placeholders(row['text'])
        if raw is None: raw=Path(row['local_image']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['image']['sha256']
        assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
        image=Image.open(io.BytesIO(raw)).convert('RGB'); image.thumbnail((1024,1024))
        prompt_text=TABLE_OCR_PROMPT if row.get('target_format')==TARGET_FORMAT else 'Transcribe all text in this image into Markdown, preserving its language, mathematical notation and reading order.'
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt_text}]}]
        args=dict(tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
        prompt=self.processor.apply_chat_template(messages,add_generation_prompt=True,**args)
        encoded=self.processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':row['text']}]}],add_generation_prompt=False,**args)
        n=prompt['input_ids'].shape[1]
        assert torch.equal(prompt['input_ids'],encoded['input_ids'][:,:n]), 'Assistant boundary mismatch'
        assert encoded['input_ids'].shape[1] <= 8192, 'Oversize sample requires explicit policy; no silent truncation'
        labels=encoded['input_ids'].clone(); labels[:,:n]=-100
        assert (labels!=-100).sum()>0
        encoded['labels']=labels
        return OCRBatch(encoded)
    def __iter__(self):
        while self.cursor<len(self.rows):
            row=self.rows[self.cursor]; self.cursor+=1
            yield self.encode(row)

class OCRDataConfig(BaseDataloaderConfig):
    kind: Literal['q35ocr_manifest']='q35ocr_manifest'
    path: str
    model_path: str
    def spawn(self): return OCRLoader(self.path,self.model_path)

class AdamConfig(OptimizerConfig):
    kind: Literal['q35ocr_adamw']='q35ocr_adamw'
    lr: float=1e-4
    def spawn(self,params): return torch.optim.AdamW(params,lr=self.lr,weight_decay=0.01)

class OCRConfig(BaseConfig):
    model: OCRModel

class OCRTrainer(BaseTrainer[OCRConfig,BaseBatch]):
    def compute_loss(self,batch):
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=self.models['model'](batch)
        assert torch.isfinite(loss), 'Nonfinite loss'
        self.last_loss=float(loss.detach())
        return loss
    def optimizer_step(self):
        # Clip without throwing locally, then coordinate the decision across
        # ranks. A single rank must not exit while its peers enter the next DDP
        # collective. This performs one health collective per optimizer update.
        norm=torch.nn.utils.clip_grad_norm_(self.get_parameters(),1.0,error_if_nonfinite=False)
        healthy=torch.isfinite(norm) & (norm>0)
        window_finite=getattr(self,'_window_finite',None)
        if window_finite is not None:
            healthy=healthy & window_finite.to(device=norm.device,dtype=torch.bool)
        healthy=healthy.to(dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(healthy,op=torch.distributed.ReduceOp.MIN)
        self._window_finite=None
        if not bool(healthy.item()):
            raise FloatingPointError('Nonfinite loss/gradient or zero gradient on at least one rank')
        super().optimizer_step()
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank()==0:
            last_loss=float(self.last_loss) if isinstance(self.last_loss,torch.Tensor) else self.last_loss
            print(json.dumps({'step':self.clock.step_index,'loss':last_loss,'gradient_norm':float(norm),'lr':self.optimizer.param_groups[0]['lr']}),flush=True)
    def adapter_state(self): return {n:p.detach().cpu().clone() for n,p in self.models['model'].named_parameters() if p.requires_grad}
    def save(self,path):
        state={'adapter':self.adapter_state(),'optimizer':self.optimizer.state_dict(),'clock':self.clock.state_dict(),'loader':self.dataloader.state_dict(),'rng':self.config.rng.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'scheduler_step':self.clock.step_index,'config':self.config.model_dump_json()}
        temp=path.with_suffix('.partial'); torch.save(state,temp); temp.replace(path)
    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        with torch.no_grad():
            for n,p in self.models['model'].named_parameters():
                if p.requires_grad: p.copy_(state['adapter'][n])
        self.optimizer.load_state_dict(state['optimizer']); self.clock.load_state_dict(state['clock']); self.dataloader.load_state_dict(state['loader']); self.config.rng.load_state_dict(state['rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.lr_scheduler.set_lr(step=state['scheduler_step'])
    def one_step(self):
        self.step(iter(self.dataloader)); self.clock.step_index+=1; self.metrics.clear()
    def _train(self):
        initial=self.adapter_state()
        self.one_step()
        assert any(not torch.equal(initial[n],p) for n,p in self.adapter_state().items()), 'No parameter update'
        checkpoint=ROOT/'checkpoints/smoke-resume.pt'; self.save(checkpoint)
        self.one_step(); expected=self.adapter_state(); loss=self.last_loss; cursor=self.dataloader.cursor
        self.restore(checkpoint); self.one_step()
        assert self.dataloader.cursor==cursor
        error=max(float((expected[n]-p).abs().max()) for n,p in self.adapter_state().items())
        assert error<1e-5 and abs(loss-self.last_loss)<1e-4, f'Resumed trajectory mismatch {error}'
        self.save(ROOT/'checkpoints/smoke-final.pt')
        os.environ['AWS_ENDPOINT_URL']=(ROOT/'endpoint.txt').read_text().strip(); s3=client()
        with checkpoint.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        key='checkpoints/smoke/smoke-resume.pt'; s3.upload_file(str(checkpoint),BUCKET,key)
        restored=checkpoint.with_name('smoke-s3-roundtrip.pt'); s3.download_file(BUCKET,key,str(restored))
        with restored.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest
        self.restore(restored)
        self.models['model'].eval()
        val=OCRLoader(self.config.data.path,self.config.data.model_path,'validation')
        with torch.no_grad():
            losses=[float(self.compute_loss(b.to(self.device))) for b in val]
        result={'status':'passed','engine':'Arsenal BaseTrainer 2026.7.11','gates':{'model_processor':True,'ocr_pairing_and_masking':True,'finite_forward_backward':True,'parameter_update':True,'checkpoint_resume_trajectory':True,'pvc_s3_roundtrip':True},'resume_parameter_max_error':error,'validation_loss':sum(losses)/len(losses),'gpu':torch.cuda.get_device_name(),'max_gpu_allocated':torch.cuda.max_memory_allocated()}
        payload=json.dumps(result,indent=2).encode(); s3.put_object(Bucket=BUCKET,Key='documentation/smoke-result.json',Body=payload)
        assert s3.get_object(Bucket=BUCKET,Key='documentation/smoke-result.json')['Body'].read()==payload
        (ROOT/'logs/smoke-result.json').write_bytes(payload)
        self.clock.stop(); print('SMOKE_ALL_GATES_PASSED',json.dumps(result),flush=True)

def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    assert json.loads((ROOT/'logs/staging-result.json').read_text())['status']=='passed'
    config=OCRConfig(run_directory=ROOT/'logs/arsenal-smoke',model=OCRModel(model_path=str(ROOT/'model')),data=OCRDataConfig(path=str(ROOT/'data/smoke.jsonl'),model_path=str(ROOT/'model'),batch_size=1),optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-4,max_lr=1e-4,min_lr_ratio=0.1,max_steps=10,use_warmup=False,warmup_steps=0,alpha=1.0),distributed=DistributedConfig(device_type='cuda',num_process_per_node=1,hsdp=HsdpConfig(replicate_size=1,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    OCRTrainer(config).train()

if __name__=='__main__': main()
