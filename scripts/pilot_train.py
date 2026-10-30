"""Bounded, isolated OCR pilot. Never launches full-data training or prunes files."""
import hashlib, heapq, json, math, os, random, statistics, time
from pathlib import Path
from typing import Literal
import torch
import torch.distributed as dist
from transformers import AutoProcessor
from arsenal.config.base import BaseConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.compile import CompileConfig
from arsenal.config.dataloader import BaseDataloaderConfig
from lr_scheduler_basic import CosineSchedulerConfig
from arsenal_ocr import OCRLoader, OCRModel, AdamConfig, ROOT as SOURCE, BUCKET
from train_full import connect
from distributed_ocr import DistributedTraining
from large_batch import AccumulationMixin

RUN=SOURCE/'pilot-10h-v1'
PREFIX='pilot-10h-v1/'
START=time.time()

def write(path,value):
    tmp=path.with_suffix('.partial'); tmp.write_text(json.dumps(value,default=str)); tmp.replace(path)

def upload(path):
    s=connect(); key=PREFIX+str(path.relative_to(RUN)); s.upload_file(str(path),BUCKET,key)
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    body=s.get_object(Bucket=BUCKET,Key=key)['Body']; h=hashlib.sha256()
    try:
        for b in body.iter_chunks(8388608):h.update(b)
    finally:body.close()
    assert h.hexdigest()==sha
    return {'path':str(path),'key':key,'sha256':sha}

def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'started.json').exists() and os.environ.get('PILOT_REUSE_PREPARED')=='1':
        assert time.time()-json.loads((RUN/'started.json').read_text())['started']<36000
        assert not (RUN/'metrics.jsonl').exists(), 'Only pre-training setup recovery allowed'
        for name in ['sampling-audit.json','source-checkpoint.json','train.jsonl','validation.jsonl','test.jsonl']:
            assert (RUN/name).is_file()
            upload(RUN/name)
        marker=json.loads((RUN/'source-checkpoint.json').read_text())
        with open(marker['path'],'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
        return
    assert not (RUN/'started.json').exists(), 'Pilot already exists: inspect before any resume'
    write(RUN/'started.json',{'started':START,'budget_seconds':36000,'batch':256,'scale_authorized':False})
    marker=json.loads((SOURCE/'checkpoints/latest.json').read_text())
    with open(marker['path'],'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
    body=connect().get_object(Bucket=BUCKET,Key=marker['key'])['Body']; h=hashlib.sha256()
    try:
        for b in body.iter_chunks(8388608):h.update(b)
    finally:body.close()
    assert h.hexdigest()==marker['sha256']
    write(RUN/'source-checkpoint.json',marker)
    reports=json.loads((SOURCE/'data/COMPLETE.json').read_text())['shards']
    chosen={'train':[],'validation':[],'test':[]}; populations={}; s=connect()
    # Per-shard deterministic hash sampling, grouped by text-length strata.
    # One representative per group avoids over-weighting augmented variants.
    for i,report in enumerate(reports):
        m=report['manifest']; raw=s.get_object(Bucket=BUCKET,Key=m['key'])['Body'].read()
        assert hashlib.sha256(raw).hexdigest()==m['sha256']
        pools={}; seen=set(); counts={}
        for line in raw.splitlines():
            row=json.loads(line); split=row['split']; group=row['group_id']
            if group in seen:continue
            seen.add(group)
            size=len(row['text']); band='short' if size<256 else 'medium' if size<2048 else 'long'
            key=(split,band); counts[str(key)]=counts.get(str(key),0)+1
            score=int(hashlib.sha256(('42:'+row['id']).encode()).hexdigest(),16)
            pool=pools.setdefault(key,[]); cap=64 if split=='train' else 2
            item=(-score,row['id'],row)
            if len(pool)<cap:heapq.heappush(pool,item)
            elif item>pool[0]:heapq.heapreplace(pool,item)
        for (split,band),pool in pools.items():
            if split in chosen:chosen[split].extend(x[2] for x in pool)
        populations[str(i)]=counts
        if i%32==0:print('PILOT_PREPARATION',i,len(reports),flush=True)
    for split,cap in [('train',65536),('validation',256),('test',12)]:
        rows=chosen[split]; random.Random(42).shuffle(rows); rows=rows[:cap]
        if split=='train':rows=rows[:len(rows)//256*256]
        assert rows
        chosen[split]=rows
        p=RUN/(split+'.jsonl'); p.write_text(''.join(json.dumps(r)+'\n' for r in rows)); upload(p)
    groups={k:{r['group_id'] for r in v} for k,v in chosen.items()}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    write(RUN/'sampling-audit.json',{'seed':42,'counts':{k:len(v) for k,v in chosen.items()},'shards':len(reports),
        'method':'bounded hash sample per shard, split and target-character length; one row per group',
        'limitations':'Length-balanced rather than population-proportional. Language/template not stratified; prepared manifest availability not assumed.',
        'population_group_counts':populations})
    upload(RUN/'sampling-audit.json'); upload(RUN/'source-checkpoint.json')

class PilotLoader(OCRLoader):
    def __init__(self,model_path):
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        self.rows=[json.loads(l) for l in (RUN/'train.jsonl').read_text().splitlines()]
        self.cursor=0; self.s3=connect()
    @property
    def count(self):return self.cursor
    def state_dict(self):return {'cursor':self.cursor,'count':self.cursor,'pilot':True}
    def load_state_dict(self,s):self.cursor=s['cursor'] if s.get('pilot') else 0
    def __iter__(self):
        rank=dist.get_rank(); world=dist.get_world_size()
        while self.cursor<len(self.rows):
            i=self.cursor; self.cursor+=1
            if i%world!=rank:continue
            row=self.rows[i]; im=row['image']; offset=im['offset']
            raw=self.s3.get_object(Bucket=BUCKET,Key=im['key'],Range=f"bytes={offset}-{offset+im['size']-1}")['Body'].read()
            yield self.encode(row,raw)

class PilotData(BaseDataloaderConfig):
    kind: Literal['q35ocr_pilot']='q35ocr_pilot'
    model_path:str
    def spawn(self):return PilotLoader(self.model_path)

class PilotConfig(BaseConfig):
    model:OCRModel

class PilotTrainer(AccumulationMixin,DistributedTraining):
    def record_window_loss(self,loss):
        finite=torch.isfinite(loss.detach()).all().to(torch.int32)
        self._window_finite=finite if self._window_finite is None else torch.minimum(self._window_finite,finite)
        self.last_loss=loss.detach()
        if not isinstance(self.loss_sum,torch.Tensor):
            self.loss_sum=torch.zeros((),device=loss.device,dtype=torch.float64)
        self.loss_sum.add_(loss.detach().to(dtype=torch.float64))
        self.loss_count+=1
    def compute_loss(self,batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=self.models['model'](batch)
        self.record_window_loss(loss)
        return loss
    def mean_accumulated_loss(self):
        """Reduce per-microstep loss statistics once after the optimizer update."""
        if self.loss_count<=0:raise RuntimeError('No microbatch losses were recorded')
        loss_sum=self.loss_sum
        if not isinstance(loss_sum,torch.Tensor):
            loss_sum=torch.tensor(float(loss_sum),device=self.device,dtype=torch.float64)
        else:
            loss_sum=loss_sum.to(device=self.device,dtype=torch.float64)
        count=torch.tensor(float(self.loss_count),device=self.device,dtype=torch.float64)
        stats=torch.stack((loss_sum,count))
        if dist.is_available() and dist.is_initialized():dist.all_reduce(stats)
        return float((stats[0]/stats[1]).item())
    def lr_scheduler_step(self):
        self.lr_scheduler.set_lr(step=1+(self.start_samples+self.pilot_samples)//8)
    def evaluate(self):
        dist.barrier(); result=[None]
        if dist.get_rank()==0:
            model=self.models['model'].module; model.eval(); total=0.; tokens=0
            with torch.no_grad():
                for line in (RUN/'validation.jsonl').read_text().splitlines():
                    row=json.loads(line); im=row['image']; off=im['offset']
                    raw=connect().get_object(Bucket=BUCKET,Key=im['key'],Range=f"bytes={off}-{off+im['size']-1}")['Body'].read()
                    batch=self.dataloader.encode(row,raw).to(self.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch)
                    assert torch.isfinite(loss), 'Nonfinite validation'
                    n=int((batch['labels'][:,1:]!=-100).sum()); total+=float(loss)*n; tokens+=n
            model.train(); result[0]=total/tokens
        dist.broadcast_object_list(result,src=0); return result[0]
    def save_pilot(self,label):
        import numpy as np
        local={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        rng=[None]*dist.get_world_size(); dist.all_gather_object(rng,local)
        if dist.get_rank()==0:
            p=RUN/f'checkpoint-{label}.pt'
            state={'adapter':self.adapter_state(),'optimizer':self.optimizer.state_dict(),'clock':self.clock.state_dict(),
                'loader':{'cursor':self.pilot_samples,'count':self.pilot_samples,'pilot':True},'rank_rng':rng,
                'world_size':4,'scheduler_step':1+(self.start_samples+self.pilot_samples)//8,'config':self.config.model_dump_json(),
                'source_samples':self.start_samples,'pilot_samples':self.pilot_samples,'elapsed':time.time()-self.started,
                'manifest_sha256':hashlib.sha256((RUN/'train.jsonl').read_bytes()).hexdigest()}
            tmp=p.with_suffix('.partial'); torch.save(state,tmp); tmp.replace(p)
            entry=upload(p); entry.update(global_step=self.clock.step_index,pilot_samples=self.pilot_samples)
            write(RUN/'latest.json',entry); upload(RUN/'latest.json')
            print('PILOT_CHECKPOINT_VERIFIED',json.dumps(entry),flush=True)
        dist.barrier()
    def _train(self):
        try:self._run_pilot()
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            write(RUN/f'failure-rank{dist.get_rank()}.json',{'time':time.time(),'error':repr(exc),'scale_authorized':False})
            # Fail the spawned process so the launcher terminates peer ranks.
            os._exit(1)
    def decision(self,value):
        values=[value if dist.get_rank()==0 else None]
        dist.broadcast_object_list(values,src=0)
        return values[0]
    def _run_pilot(self):
        self.started=json.loads((RUN/'started.json').read_text())['started']
        marker=json.loads((RUN/'source-checkpoint.json').read_text())
        self.start_samples=marker['samples']; self.pilot_samples=0
        self.restore(Path(marker['path'])); self.sync_check()
        best=self.evaluate(); baseline=best; bad=0; history=[]; reason='sample_limit'; best_path=marker['path']
        self.record({'pilot_update':0,'global_step':self.clock.step_index,'eval_loss':best,'learning_rate':self.optimizer.param_groups[0]['lr'],'pilot_samples':0})
        iterator=iter(self.dataloader); self.config.gradient_accumulation.count=64
        max_seconds=200.; last_save=time.time()
        for update in range(1,len(self.dataloader.rows)//256+1):
            elapsed=time.time()-self.started
            if self.decision(elapsed+max_seconds+600>=9*3600):reason='time_budget';break
            begin=time.time(); self.loss_sum=0.; self.loss_count=0
            self.step(iterator); self.clock.step_index+=1; self.metrics.clear(); self.pilot_samples+=256
            loss=self.mean_accumulated_loss(); history.append(loss)
            entry={'pilot_update':update,'global_step':self.clock.step_index,'train_loss':loss,'learning_rate':self.optimizer.param_groups[0]['lr'],'pilot_samples':self.pilot_samples}
            stop=None
            if len(history)>=13 and all(x>3*statistics.median(history[-13:-3]) for x in history[-3:]):stop='divergent_loss'
            if len(history)>=20 and statistics.pstdev(history[-10:])>statistics.mean(history[-10:]) and statistics.mean(history[-10:])>1.5*statistics.median(history[-20:-10]):stop='unstable_loss'
            if self.decision(update==1 or update%5==0 or time.time()-last_save>=600 or bool(stop)):
                score=self.evaluate(); entry['eval_loss']=score
                self.save_pilot(str(self.clock.step_index)); last_save=time.time()
                if score<best:
                    best=score; best_path=str(RUN/f'checkpoint-{self.clock.step_index}.pt')
                    if dist.get_rank()==0:
                        write(RUN/'best.json',{'path':best_path,'eval_loss':best});upload(RUN/'best.json')
                bad=bad+1 if score>best*1.2 and score-best>0.05 else 0
                if bad>=2:stop='validation_degradation'
            self.record(entry); max_seconds=max(max_seconds,time.time()-begin)
            if stop:reason=stop;break
        self.save_pilot('final-'+str(self.clock.step_index))
        if dist.get_rank()==0:
            write(RUN/'result.json',{'status':'training_stopped','reason':reason,'duration_seconds':time.time()-self.started,
                'pilot_samples':self.pilot_samples,'global_step':self.clock.step_index,'baseline_eval_loss':baseline,'best_eval_loss':best,
                'best_checkpoint':best_path,'inference_status':'pending','scale_authorized':False})
            upload(RUN/'result.json')
        dist.barrier(); self.clock.stop()
    def record(self,entry):
        entry.update(timestamp=time.time(),elapsed_seconds=time.time()-self.started)
        if dist.get_rank()==0:
            with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            write(RUN/'status.json',entry); upload(RUN/'metrics.jsonl');upload(RUN/'status.json')
            print('PILOT_METRIC',json.dumps(entry),flush=True)

def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3')
    os.environ['TRAIN_MODE']='train'
    prepare()
    config=PilotConfig(run_directory=RUN/'arsenal',model=OCRModel(model_path=str(SOURCE/'model')),
        data=PilotData(model_path=str(SOURCE/'model'),batch_size=1),optimizer=AdamConfig(),
        lr_scheduler=CosineSchedulerConfig(init_lr=1e-6,max_lr=1e-4,min_lr_ratio=.1,max_steps=460385,use_warmup=True,warmup_steps=500,alpha=1.),
        distributed=DistributedConfig(device_type='cuda',num_process_per_node=4,hsdp=HsdpConfig(replicate_size=4,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    PilotTrainer(config).train()

if __name__=='__main__':main()
