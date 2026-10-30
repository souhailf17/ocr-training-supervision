"""One deterministic full epoch using verified MinIO manifests and Arsenal."""
import hashlib, json, math, os, random, time
from pathlib import Path
from typing import Literal
import torch
from transformers import AutoProcessor
from arsenal.config.base import BaseConfig
from arsenal.config.compile import CompileConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.dataloader import BaseDataloaderConfig
from lr_scheduler_basic import CosineSchedulerConfig
from minio_retry import read_object
from arsenal_ocr import OCRLoader, OCRTrainer, OCRConfig, OCRModel, AdamConfig, ROOT, BUCKET, client
from stage_training import DATA_DIR

def connect(*, streaming=False):
    os.environ['AWS_ENDPOINT_URL']=(ROOT/'endpoint.txt').read_text().strip()
    return client(streaming=streaming)

class StreamingLoader(OCRLoader):
    def __init__(self, model_path):
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        self.s3=connect(streaming=True)
        self.reports=json.loads((DATA_DIR/'COMPLETE.json').read_text())['shards']
        random.Random(42).shuffle(self.reports)
        self.shard=0; self.cursor=0; self.rows=None; self.count=0
    def state_dict(self): return {'shard':self.shard,'cursor':self.cursor,'count':self.count}
    def load_state_dict(self,state):
        self.shard=state['shard']; self.cursor=state['cursor']; self.count=state['count']; self.rows=None
    def __iter__(self):
        while self.shard<len(self.reports):
            if self.rows is None:
                report=self.reports[self.shard]['manifest']
                raw=read_object(self.s3,BUCKET,report['key'])
                assert hashlib.sha256(raw).hexdigest()==report['sha256']
                self.rows=[r for line in raw.splitlines() if (r:=json.loads(line))['split']=='train']
                random.Random(42+self.shard).shuffle(self.rows)
            while self.cursor<len(self.rows):
                row=self.rows[self.cursor]; im=row['image']; start=im['offset']
                raw=read_object(self.s3,BUCKET,im['key'],byte_range=f"bytes={start}-{start+im['size']-1}")
                batch=self.encode(row,raw)
                self.cursor+=1; self.count+=1
                yield batch
            self.shard+=1; self.cursor=0; self.rows=None

class StreamingConfig(BaseDataloaderConfig):
    kind: Literal['q35ocr_stream']='q35ocr_stream'
    model_path: str
    def spawn(self): return StreamingLoader(self.model_path)

class FullConfig(BaseConfig):
    model: OCRModel

class FullTrainer(OCRTrainer):
    def evaluate(self):
        if not hasattr(self,'validation_rows'):
            self.validation_rows=[]; groups=set(); s3=connect(streaming=True)
            reports=json.loads((DATA_DIR/'COMPLETE.json').read_text())['shards']
            # Two validation groups from each of 32 evenly spaced shards.
            for idx in [i*(len(reports)-1)//31 for i in range(32)]:
                report=reports[idx]['manifest']; data=read_object(s3,BUCKET,report['key'])
                assert hashlib.sha256(data).hexdigest()==report['sha256']
                selected=0
                for line in data.splitlines():
                    row=json.loads(line)
                    if row['split']!='validation' or row['group_id'] in groups: continue
                    self.validation_rows.append(row); groups.add(row['group_id']); selected+=1
                    if selected==2: break
            assert len(self.validation_rows)==64
            (DATA_DIR/'monitor-validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.validation_rows))
        model=self.models['model']; model.eval(); total=0.; tokens=0; s3=connect(streaming=True)
        with torch.no_grad():
            for row in self.validation_rows:
                im=row['image']; start=im['offset']; raw=read_object(s3,BUCKET,im['key'],byte_range=f"bytes={start}-{start+im['size']-1}")
                batch=self.dataloader.encode(row,raw).to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(batch)
                assert torch.isfinite(loss)
                n=int((batch['labels'][:,1:]!=-100).sum()); total+=float(loss)*n; tokens+=n
        model.train()
        result={'step':self.clock.step_index,'validation_loss':total/tokens,'validation_samples':64,'supervised_tokens':tokens}
        with (ROOT/'logs/validation.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
        print('VALIDATION',json.dumps(result),flush=True)
    def checkpoint(self, final=False):
        step=self.clock.step_index
        path=ROOT/'checkpoints'/f'train-{step:08d}.pt'
        self.save(path)
        s3=connect(); key='training/checkpoints/'+path.name
        s3.upload_file(str(path),BUCKET,key)
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        body=s3.get_object(Bucket=BUCKET,Key=key)['Body']; h=hashlib.sha256()
        try:
            for chunk in body.iter_chunks(8*1024*1024): h.update(chunk)
        finally: body.close()
        assert h.hexdigest()==digest
        marker={'path':str(path),'key':key,'sha256':digest,'step':step,'samples':self.dataloader.count,'complete':final}
        temp=ROOT/'checkpoints/latest.partial'; temp.write_text(json.dumps(marker)); temp.replace(ROOT/'checkpoints/latest.json')
        s3.put_object(Bucket=BUCKET,Key='training/checkpoints/latest.json',Body=json.dumps(marker).encode())
        # Retain three local task checkpoints, all already verified in MinIO.
        for old in sorted((ROOT/'checkpoints').glob('train-*.pt'))[:-3]: old.unlink()
        for log in (ROOT/'logs').glob('*'):
            if log.is_file(): s3.upload_file(str(log),BUCKET,'training/logs/'+log.name)
    def _train(self):
        marker=ROOT/'checkpoints/latest.json'
        if marker.exists():
            saved=json.loads(marker.read_text()); path=Path(saved['path'])
            with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==saved['sha256']
            self.restore(path)
        self.evaluate()
        started=time.time(); iterator=iter(self.dataloader)
        while True:
            # Materialize up to eight batches so the final partial update is
            # scaled correctly instead of silently dropping its gradients.
            batches=[]
            for _ in range(8):
                try: batches.append(next(iterator))
                except StopIteration: break
            if not batches: break
            self.config.gradient_accumulation.count=len(batches)
            self.step(iter(batches)); self.clock.step_index+=1; self.metrics.clear()
            status={'state':'running','step':self.clock.step_index,'samples':self.dataloader.count,'loss':self.last_loss,'elapsed_this_run_seconds':time.time()-started,'gpu_memory_bytes':torch.cuda.max_memory_allocated()}
            (ROOT/'logs/training-status.json').write_text(json.dumps(status))
            with (ROOT/'logs/metrics.jsonl').open('a') as f: f.write(json.dumps(status)+'\n')
            if self.dataloader.count<=8 or self.clock.step_index%250==0: self.checkpoint()
            if self.clock.step_index%1000==0: self.evaluate()
        self.evaluate(); self.checkpoint(final=True)
        (ROOT/'logs/training-complete.json').write_text(json.dumps({'state':'complete','step':self.clock.step_index,'samples':self.dataloader.count}))
        self.clock.stop()

def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    result=json.loads((ROOT/'logs/smoke-result.json').read_text())
    assert result['status']=='passed' and all(result['gates'].values()), 'Smoke gates not passed'
    s3=connect()
    assert json.loads(s3.get_object(Bucket=BUCKET,Key='documentation/smoke-result.json')['Body'].read())==result
    count=json.loads((ROOT/'data/COMPLETE.json').read_text())['counts']['train']
    steps=math.ceil(count/8)
    config=FullConfig(run_directory=ROOT/'logs/arsenal-training',model=OCRModel(model_path=str(ROOT/'model')),data=StreamingConfig(model_path=str(ROOT/'model'),batch_size=1),optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-6,max_lr=1e-4,min_lr_ratio=0.1,max_steps=steps,use_warmup=True,warmup_steps=500,alpha=1.0),distributed=DistributedConfig(device_type='cuda',num_process_per_node=1,hsdp=HsdpConfig(replicate_size=1,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
    FullTrainer(config).train()

if __name__=='__main__': main()
