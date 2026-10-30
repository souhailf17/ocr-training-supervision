"""Global batch 16384 with bounded image microbatches and sample-based LR."""
import contextlib,hashlib,json,math,os,time
from pathlib import Path
from typing import Literal
import torch
import torch.distributed as dist
from arsenal.config.base import BaseConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.compile import CompileConfig
from lr_scheduler_basic import CosineSchedulerConfig
from arsenal_ocr import OCRBatch,OCRModel,AdamConfig,ROOT,BUCKET,client
from distributed_ocr import RankSmokeData,RankStreamConfig,DistributedSmoke,DistributedTraining

MICRO=int(os.environ.get('MICROBATCH','2'))
RANKS=int(os.environ.get('TRAIN_RANKS','4'))
GLOBAL=16384
TOTAL=3683080
SMOKE_KEY='documentation/large-batch-smoke-result.json'

def collate(rows,pad):
    result={}; width=max(r['input_ids'].shape[1] for r in rows)
    for key in rows[0]:
        values=[r[key] for r in rows]
        if key in ['input_ids','attention_mask','labels','mm_token_type_ids','token_type_ids']:
            fill=-100 if key=='labels' else pad if key=='input_ids' else 0
            values=[torch.nn.functional.pad(v,(0,width-v.shape[1]),value=fill) for v in values]
        elif key not in ['pixel_values','image_grid_thw']:
            raise ValueError('Unvalidated processor batch field: '+key)
        result[key]=torch.cat(values,dim=0)
    return OCRBatch(result)

class BatchedLoader:
    def __init__(self,inner):self.inner=inner; self.remaining=None
    @property
    def count(self):return getattr(self.inner,'count',self.inner.cursor)
    @property
    def cursor(self):return self.inner.cursor
    def state_dict(self):return self.inner.state_dict()
    def load_state_dict(self,state):self.inner.load_state_dict(state)
    def encode(self,*args,**kw):return self.inner.encode(*args,**kw)
    def __iter__(self):
        iterator=iter(self.inner)
        while True:
            rows=[]
            size=MICRO if self.remaining is None else min(MICRO,self.remaining)
            if size==0:return
            for _ in range(size):
                try:rows.append(next(iterator))
                except StopIteration:break
            if not rows:return
            if self.remaining is not None:self.remaining-=len(rows)
            yield collate(rows,self.inner.processor.tokenizer.pad_token_id)

class LargeSmokeData(RankSmokeData):
    kind: Literal['q35ocr_large_smoke']='q35ocr_large_smoke'
    def spawn(self):
        inner=super().spawn(); inner.rows*=4
        return BatchedLoader(inner)

class LargeStreamData(RankStreamConfig):
    kind: Literal['q35ocr_large_stream']='q35ocr_large_stream'
    def spawn(self):return BatchedLoader(super().spawn())

class LargeConfig(BaseConfig):
    model: OCRModel

class AccumulationMixin:
    def microstep_ctx(self,batch):
        self._micro_index+=1
        return self.models['model'].no_sync() if self._micro_index<self.config.gradient_accumulation.count else contextlib.nullcontext()
    def step(self,iterator):
        self._micro_index=0
        self._window_finite=None
        interval=1 if os.environ.get('TRAIN_MODE','smoke')=='smoke' else int(os.environ.get('Q35OCR_RANK_SYNC_INTERVAL','16'))
        if interval<1:raise ValueError('Q35OCR_RANK_SYNC_INTERVAL must be positive')
        def synchronized():
            for index in range(self.config.gradient_accumulation.count):
                if index%interval==0:
                    # Preserve the proven rank-skew guard until matched H200
                    # profiling demonstrates that removing this sync is safe.
                    torch.cuda.synchronize()
                    dist.barrier()
                yield next(iterator)
        return super().step(synchronized())

class LargeSmoke(AccumulationMixin,DistributedSmoke):
    def _train(self):
        self.config.gradient_accumulation.count=2
        super()._train()
        if dist.get_rank()==0:
            from train_full import connect
            s3=connect(); result=json.loads(s3.get_object(Bucket=BUCKET,Key='documentation/ddp-smoke-result.json')['Body'].read())
            result.update({'microbatch_per_gpu':MICRO,'tested_accumulation':2,'configured_global_batch':GLOBAL})
            raw=json.dumps(result).encode(); s3.put_object(Bucket=BUCKET,Key=SMOKE_KEY,Body=raw)
            assert s3.get_object(Bucket=BUCKET,Key=SMOKE_KEY)['Body'].read()==raw
        dist.barrier()

class LargeTraining(AccumulationMixin,DistributedTraining):
    def compute_loss(self,batch):
        loss=super().compute_loss(batch)
        # Arsenal divides by accumulation count. Weight the final short
        # microbatch proportionally so each example has the same weight.
        if hasattr(self,'_examples_per_rank'):
            loss=loss*(batch['input_ids'].shape[0]*self.config.gradient_accumulation.count/self._examples_per_rank)
        return loss
    def lr_scheduler_step(self):
        # Preserve the pre-migration schedule in units of eight examples.
        self.lr_scheduler.set_lr(step=1+self._update_start_samples//8)
    def _train(self):
        marker=json.loads((ROOT/'checkpoints/latest.json').read_text()); path=Path(marker['path'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
        if dist.get_rank()==0:
            from train_full import connect
            body=connect().get_object(Bucket=BUCKET,Key=marker['key'])['Body']; digest=hashlib.sha256()
            try:
                for chunk in body.iter_chunks(8388608):digest.update(chunk)
            finally:body.close()
            assert digest.hexdigest()==marker['sha256'], 'S3 checkpoint checksum mismatch'
            print('RESUME_CHECKPOINT_VERIFIED',json.dumps(marker),flush=True)
        dist.barrier()
        self.restore(path); self.sync_check(); self.evaluate()
        consumed=marker['samples']; initial=consumed; started=time.time()
        while consumed<TOTAL:
            count=min(GLOBAL,TOTAL-consumed)
            assert count%dist.get_world_size()==0, 'Unequal final rank counts'
            self._examples_per_rank=count//dist.get_world_size()
            accumulation=math.ceil(self._examples_per_rank/MICRO)
            self.dataloader.remaining=self._examples_per_rank
            # A BatchedLoader iterator is exhausted at each global update
            # because ``remaining`` deliberately bounds that update. Create a
            # fresh iterator so the next update continues from the restored
            # global cursor instead of immediately raising StopIteration.
            iterator=iter(self.dataloader)
            self.config.gradient_accumulation.count=accumulation
            self._update_start_samples=consumed
            def tracked():
                for index in range(accumulation):
                    # Bound rank skew across long no_sync accumulation windows.
                    # This synchronizes progress, not gradients or optimizer steps.
                    if index%16==0:
                        progress={'rank':dist.get_rank(),'step':self.clock.step_index,'microsteps_completed':index,'accumulation':accumulation,'time':time.time()}
                        target=ROOT/'logs'/f'progress-rank{dist.get_rank()}.json'
                        temporary=target.with_suffix('.partial')
                        temporary.write_text(json.dumps(progress)); temporary.replace(target)
                    batch=next(iterator)
                    if dist.get_rank()==0 and index%16==0:
                        status={'state':'accumulating_gradients','world_size':RANKS,'global_batch_size':GLOBAL,'microbatch_per_gpu':MICRO,'gradient_accumulation':accumulation,'microsteps_completed':index,'step':self.clock.step_index,'samples':consumed+index*MICRO*RANKS,'samples_at_last_update':consumed,'elapsed_this_run_seconds':time.time()-started}
                        (ROOT/'logs/training-status.json').write_text(json.dumps(status))
                    yield batch
            self.step(tracked()); self.clock.step_index+=1; self.metrics.clear(); consumed+=count
            observed=torch.tensor(self.dataloader.count,device=self.device); dist.all_reduce(observed,op=dist.ReduceOp.MAX)
            assert int(observed)==consumed
            self.checkpoint(final=consumed==TOTAL)
            if dist.get_rank()==0:
                status={'state':'running','world_size':RANKS,'global_batch_size':GLOBAL,'microbatch_per_gpu':MICRO,'gradient_accumulation':accumulation,'step':self.clock.step_index,'samples':consumed,'samples_per_second':(consumed-initial)/(time.time()-started),'lr':self.optimizer.param_groups[0]['lr']}
                (ROOT/'logs/training-status.json').write_text(json.dumps(status))
                with (ROOT/'logs/metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            self.evaluate()
        if dist.get_rank()==0:(ROOT/'logs/training-complete.json').write_text(json.dumps({'state':'complete','samples':consumed,'global_batch_size':GLOBAL}))
        dist.barrier(); self.clock.stop()

def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',','.join(map(str,range(RANKS)))); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    mode=os.environ.get('TRAIN_MODE','smoke')
    if mode=='train':
        os.environ['AWS_ENDPOINT_URL']='http://q35ocr-20260921-p560-s3-relay.mtrna-ns1.svc:9000'
        result=json.loads(client().get_object(Bucket=BUCKET,Key=SMOKE_KEY)['Body'].read())
        assert result['status']=='passed' and result['microbatch_per_gpu']==MICRO and len(result['ranks'])==RANKS
    data=LargeStreamData(model_path=str(ROOT/'model'),batch_size=MICRO) if mode=='train' else LargeSmokeData(path=str(ROOT/'data/smoke.jsonl'),model_path=str(ROOT/'model'),batch_size=MICRO)
    config=LargeConfig(run_directory=ROOT/'logs/arsenal-large-batch',model=OCRModel(model_path=str(ROOT/'model')),data=data,optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-6 if mode=='train' else 1e-4,max_lr=1e-4,min_lr_ratio=.1,max_steps=460385 if mode=='train' else 10,use_warmup=mode=='train',warmup_steps=500 if mode=='train' else 0,alpha=1.),distributed=DistributedConfig(device_type='cuda',num_process_per_node=RANKS,hsdp=HsdpConfig(replicate_size=RANKS,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    (LargeTraining if mode=='train' else LargeSmoke)(config).train()

if __name__=='__main__':main()
