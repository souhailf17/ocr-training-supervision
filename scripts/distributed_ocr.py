"""Four-rank Arsenal data parallel smoke and resumable training."""
import json, os, time, hashlib
from pathlib import Path
from typing import Literal
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from arsenal_ocr import OCRTrainer, OCRConfig, OCRModel, OCRDataConfig, OCRLoader, AdamConfig, ROOT, BUCKET, client
from minio_retry import read_object
from train_full import FullTrainer, StreamingConfig, StreamingLoader, connect
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.compile import CompileConfig
from arsenal.config.base import BaseConfig
from lr_scheduler_basic import CosineSchedulerConfig

class RankSmokeData(OCRDataConfig):
    kind: Literal['q35ocr_rank_smoke']='q35ocr_rank_smoke'
    def spawn(self):
        loader=super().spawn()
        loader.rows=loader.rows[dist.get_rank()::dist.get_world_size()]
        assert len(loader.rows)>=2
        return loader

class RankStream(StreamingLoader):
    def __iter__(self):
        # All ranks advance the same deterministic cursor. Only the rank owning
        # each sample fetches/encodes the image, so no training sample is repeated.
        import random
        while self.shard<len(self.reports):
            if self.rows is None:
                report=self.reports[self.shard]['manifest']
                raw=read_object(self.s3,BUCKET,report['key'])
                assert hashlib.sha256(raw).hexdigest()==report['sha256']
                self.rows=[r for line in raw.splitlines() if (r:=json.loads(line))['split']=='train']
                random.Random(42+self.shard).shuffle(self.rows)
            while self.cursor<len(self.rows):
                row=self.rows[self.cursor]
                owner=self.count%dist.get_world_size()
                if owner!=dist.get_rank():
                    self.cursor+=1; self.count+=1
                    continue
                im=row['image']; start=im['offset']
                raw=read_object(self.s3,BUCKET,im['key'],byte_range=f"bytes={start}-{start+im['size']-1}")
                batch=self.encode(row,raw)
                # Commit this owned example only after a successful full read.
                self.cursor+=1; self.count+=1
                yield batch
            self.shard+=1; self.cursor=0; self.rows=None

class RankStreamConfig(StreamingConfig):
    kind: Literal['q35ocr_rank_stream']='q35ocr_rank_stream'
    def spawn(self): return RankStream(self.model_path)

class DDPConfig(BaseConfig):
    model: OCRModel

class DistributedMixin:
    def setup(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        super().setup()
    def materialize_models(self):
        super().materialize_models()
        self.models['model']=DDP(self.models['model'],device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    def sync_check(self):
        probe=next(p for p in self.get_parameters() if p.numel()>100).detach().flatten()[:100].clone()
        low=probe.clone(); high=probe.clone()
        dist.all_reduce(low,op=dist.ReduceOp.MIN); dist.all_reduce(high,op=dist.ReduceOp.MAX)
        assert torch.equal(low,high), 'Ranks diverged'

class DistributedSmoke(DistributedMixin,OCRTrainer):
    def _train(self):
        rank=dist.get_rank()
        initial=self.adapter_state(); begin=time.time()
        self.one_step()
        assert any(not torch.equal(initial[n],p) for n,p in self.adapter_state().items())
        self.sync_check()
        path=ROOT/'checkpoints'/f'ddp-smoke-rank{rank}.pt'; self.save(path)
        self.one_step(); expected=self.adapter_state(); loss=self.last_loss
        self.restore(path); self.one_step()
        error=max(float((expected[n]-p).abs().max()) for n,p in self.adapter_state().items())
        print('RESUME_DIAGNOSTIC',rank,error,abs(loss-self.last_loss),flush=True)
        assert error<1e-5 and abs(loss-self.last_loss)<1e-4
        self.sync_check()
        s3=connect(); key=f'training/ddp-smoke/{path.name}'; s3.upload_file(str(path),BUCKET,key)
        body=s3.get_object(Bucket=BUCKET,Key=key)['Body']; h=hashlib.sha256()
        for chunk in body.iter_chunks(8*1024*1024): h.update(chunk)
        body.close()
        with path.open('rb') as f: assert h.hexdigest()==hashlib.file_digest(f,'sha256').hexdigest()
        result={'rank':rank,'world_size':dist.get_world_size(),'status':'passed','resume_max_error':error,'gpu':torch.cuda.get_device_name(),'elapsed_seconds':time.time()-begin,'peak_memory':torch.cuda.max_memory_allocated()}
        gathered=[None]*dist.get_world_size(); dist.all_gather_object(gathered,result)
        if rank==0:
            payload=json.dumps({'status':'passed','gates':['distinct_rank_data','finite_gradients','parameter_updates','rank_synchronization','exact_resume','pvc_s3_roundtrip'],'ranks':gathered}).encode()
            s3.put_object(Bucket=BUCKET,Key='documentation/ddp-smoke-result.json',Body=payload)
            assert s3.get_object(Bucket=BUCKET,Key='documentation/ddp-smoke-result.json')['Body'].read()==payload
            (ROOT/'logs/ddp-smoke-result.json').write_bytes(payload)
            print('DISTRIBUTED_SMOKE_PASSED',payload.decode(),flush=True)
        dist.barrier(); self.clock.stop()

class DistributedTraining(DistributedMixin,FullTrainer):
    def evaluate(self):
        dist.barrier()
        if dist.get_rank()==0:
            wrapped=self.models['model']; self.models['model']=wrapped.module
            try: super().evaluate()
            finally: self.models['model']=wrapped
        dist.barrier()
    def restore(self,path):
        import random,numpy as np
        state=torch.load(path,map_location='cpu',weights_only=False)
        with torch.no_grad():
            for n,p in self.models['model'].named_parameters():
                if p.requires_grad:
                    p.copy_(state['adapter'].get(n,state['adapter'].get(n.removeprefix('module.'))))
        self.optimizer.load_state_dict(state['optimizer']); self.clock.load_state_dict(state['clock'])
        self.dataloader.load_state_dict(state['loader'])
        rankstate=state.get('rank_rng',[state]*dist.get_world_size())[dist.get_rank()]
        random.setstate(rankstate['python_rng']); np.random.set_state(rankstate['numpy_rng']); torch.set_rng_state(rankstate['torch_rng'])
        cuda=rankstate['cuda_rng']; torch.cuda.set_rng_state(cuda[0] if isinstance(cuda,list) else cuda)
        self.lr_scheduler.set_lr(step=state['scheduler_step'])
    def checkpoint(self,final=False):
        import random,numpy as np
        rank=dist.get_rank(); count=torch.tensor(self.dataloader.count,device=self.device)
        states=[None]*dist.get_world_size(); dist.all_gather_object(states,self.dataloader.state_dict())
        canonical=max(states,key=lambda s:s['count'])
        assert canonical['count']%dist.get_world_size()==0
        local={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        rng=[None]*dist.get_world_size(); dist.all_gather_object(rng,local)
        if rank==0:
            step=self.clock.step_index; path=ROOT/'checkpoints'/f'ddp-train-{step:08d}.pt'
            state={'adapter':self.adapter_state(),'optimizer':self.optimizer.state_dict(),'clock':self.clock.state_dict(),'loader':canonical,'rank_rng':rng,'world_size':dist.get_world_size(),'scheduler_step':step,'config':self.config.model_dump_json()}
            tmp=path.with_suffix('.partial'); torch.save(state,tmp); tmp.replace(path)
            s3=connect(); key='training/checkpoints/'+path.name; s3.upload_file(str(path),BUCKET,key)
            with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            body=s3.get_object(Bucket=BUCKET,Key=key)['Body']; h=hashlib.sha256()
            for chunk in body.iter_chunks(8*1024*1024):h.update(chunk)
            body.close(); assert h.hexdigest()==digest
            marker={'path':str(path),'key':key,'sha256':digest,'step':step,'samples':canonical['count'],'complete':final,'world_size':dist.get_world_size()}
            tmp=ROOT/'checkpoints/latest.partial'; tmp.write_text(json.dumps(marker)); tmp.replace(ROOT/'checkpoints/latest.json')
            s3.put_object(Bucket=BUCKET,Key='training/checkpoints/latest.json',Body=json.dumps(marker).encode())
            for old in sorted((ROOT/'checkpoints').glob('ddp-train-*.pt'))[:-3]:old.unlink()
            for log in (ROOT/'logs').glob('*'):
                if log.is_file():s3.upload_file(str(log),BUCKET,'training/logs/'+log.name)
            print('DISTRIBUTED_CHECKPOINT_VERIFIED',json.dumps(marker),flush=True)
        dist.barrier()
    def _train(self):
        marker=json.loads((ROOT/'checkpoints/latest.json').read_text()); path=Path(marker['path'])
        with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
        self.restore(path); self.sync_check(); self.evaluate()
        start=time.time(); initial=marker['samples']; iterator=iter(self.dataloader); updates=0
        while True:
            batches=[]
            for _ in range(2):
                try:batches.append(next(iterator))
                except StopIteration:break
            n=torch.tensor(len(batches),device=self.device); lo=n.clone(); hi=n.clone()
            dist.all_reduce(lo,op=dist.ReduceOp.MIN); dist.all_reduce(hi,op=dist.ReduceOp.MAX)
            assert lo==hi, 'Unequal rank batch counts; do not drop samples'
            if n==0:break
            self.config.gradient_accumulation.count=len(batches)
            self.step(iter(batches)); self.clock.step_index+=1; self.metrics.clear(); updates+=1
            count=torch.tensor(self.dataloader.count,device=self.device); dist.all_reduce(count,op=dist.ReduceOp.MAX)
            if dist.get_rank()==0:
                status={'state':'running','world_size':dist.get_world_size(),'step':self.clock.step_index,'samples':int(count),'loss_rank0_last_microbatch':self.last_loss,'elapsed_this_run_seconds':time.time()-start,'samples_per_second':(int(count)-initial)/(time.time()-start),'gpu_memory_bytes_per_rank':torch.cuda.max_memory_allocated()}
                (ROOT/'logs/training-status.json').write_text(json.dumps(status))
                with (ROOT/'logs/metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            if updates==1 or self.clock.step_index%250==0:self.checkpoint()
            if self.clock.step_index%1000==0:self.evaluate()
        self.evaluate(); self.checkpoint(final=True)
        if dist.get_rank()==0:(ROOT/'logs/training-complete.json').write_text(json.dumps({'state':'complete','samples':self.dataloader.count,'world_size':dist.get_world_size()}))
        dist.barrier(); self.clock.stop()

def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    ranks=int(os.environ.get('TRAIN_RANKS','4'))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',','.join(map(str,range(ranks))))
    mode=os.environ.get('TRAIN_MODE','smoke')
    cls=DistributedSmoke
    data=RankSmokeData(path=str(ROOT/'data/smoke.jsonl'),model_path=str(ROOT/'model'),batch_size=1)
    config=DDPConfig(run_directory=ROOT/'logs/arsenal-ddp-smoke',model=OCRModel(model_path=str(ROOT/'model')),data=data,optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-4,max_lr=1e-4,min_lr_ratio=.1,max_steps=10,use_warmup=False,warmup_steps=0,alpha=1.),distributed=DistributedConfig(device_type='cuda',num_process_per_node=ranks,hsdp=HsdpConfig(replicate_size=ranks,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    if mode=='train':
        os.environ['AWS_ENDPOINT_URL']='http://q35ocr-20260921-p560-s3-relay.mtrna-ns1.svc:9000'
        (ROOT/'endpoint.txt').write_text(os.environ['AWS_ENDPOINT_URL'])
        result=json.loads(client().get_object(Bucket=BUCKET,Key='documentation/ddp-smoke-result.json')['Body'].read())
        assert result['status']=='passed' and len(result['ranks'])==ranks
        cls=DistributedTraining
        config=DDPConfig(run_directory=ROOT/'logs/arsenal-ddp-training',model=OCRModel(model_path=str(ROOT/'model')),data=RankStreamConfig(model_path=str(ROOT/'model'),batch_size=1),optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-6,max_lr=1e-4,min_lr_ratio=.1,max_steps=460385,use_warmup=True,warmup_steps=500,alpha=1.),distributed=DistributedConfig(device_type='cuda',num_process_per_node=ranks,hsdp=HsdpConfig(replicate_size=ranks,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    cls(config).train()

if __name__=='__main__': main()
