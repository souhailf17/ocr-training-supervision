"""Isolated four-rank Arsenal benchmark. Never advances production checkpoints."""
import os,time,json,hashlib,threading,subprocess,resource
from pathlib import Path
from typing import Literal
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TRAIN_MODE']='train'
import torch
import torch.distributed as dist
import arsenal_ocr
from arsenal_ocr import OCRTrainer,OCRModel,AdamConfig,ROOT as SOURCE,BUCKET
from pilot_train import PilotLoader,PilotTrainer,PilotConfig
from distributed_ocr import DistributedTraining
from arsenal.config.dataloader import BaseDataloaderConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.compile import CompileConfig
from lr_scheduler_basic import CosineSchedulerConfig
from ordered_prefetch import OrderedPrefetch
from train_full import connect
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
MODE=os.environ.get('BENCH_MODE','fast')
OUT=SOURCE/'optimization'/('benchmark-'+MODE+'-v3')
CALLS={}
if MODE!='baseline':
 assert qwen.is_fast_path_available, 'Optimized kernel unavailable'
 for name in ['causal_conv1d_fn','chunk_gated_delta_rule']:
  fn=getattr(qwen,name)
  def wrapped(*args,_fn=fn,_name=name,**kwargs):
   CALLS[_name]=CALLS.get(_name,0)+1
   with torch.profiler.record_function(_name):return _fn(*args,**kwargs)
  setattr(qwen,name,wrapped)
else:assert not qwen.is_fast_path_available

class BenchLoader(PilotLoader):
 def __init__(self,model_path):
  super().__init__(model_path);self.read_seconds=0.;self.encode_seconds=0.
  self.rank=dist.get_rank();self.world=dist.get_world_size()
 def __iter__(self):
  while self.cursor<1280:
   i=self.cursor;self.cursor+=1
   if i%self.world!=self.rank:continue
   row=self.rows[i];im=row['image'];off=im['offset'];t=time.perf_counter()
   body=self.s3.get_object(Bucket=BUCKET,Key=im['key'],Range=f"bytes={off}-{off+im['size']-1}")['Body']
   try:raw=body.read()
   finally:body.close()
   self.read_seconds+=time.perf_counter()-t;t=time.perf_counter()
   batch=self.encode(row,raw);self.encode_seconds+=time.perf_counter()-t
   if MODE=='prefetch':batch=type(batch)({k:v.pin_memory() for k,v in batch.items()})
   yield batch
class BenchData(BaseDataloaderConfig):
 kind:Literal['q35ocr_perf_bench']='q35ocr_perf_bench'
 model_path:str
 def spawn(self):
  loader=BenchLoader(self.model_path)
  return OrderedPrefetch(loader,depth=4) if MODE=='prefetch' else loader

from arsenal.config.base import BaseConfig
class BenchConfig(BaseConfig):
 model:OCRModel

class BenchTrainer(PilotTrainer):
 def setup(self):
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
  arsenal_ocr.ROOT=OUT
  super().setup()
 def lr_scheduler_step(self):self.lr_scheduler.set_lr(step=1)
 def event_start(self,name):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();self.events.append((name,a,b));return b
 def batch_get(self,iterator):
  t=time.perf_counter();batch=next(iterator);self.wait_seconds+=time.perf_counter()-t
  self.sequences.append(batch['input_ids'].shape[1]);self.labels+=int((batch['labels']!=-100).sum())
  if 'image_grid_thw' in batch:self.visual+=int(batch['image_grid_thw'].prod(dim=-1).sum())//4
  event=self.event_start('h2d');batch=batch.to(self.device,non_blocking=True);event.record();return batch
 def compute_loss(self,batch):
  event=self.event_start('forward');loss=super().compute_loss(batch);event.record();return loss
 def backward(self,loss):
  event=self.event_start('backward_including_gradient_sync');super().backward(loss);event.record()
 def optimizer_step(self):
  event=self.event_start('optimizer');super().optimizer_step();event.record()
 def run_update(self,index,iterator):
  self.events=[];self.wait_seconds=0.;self.sequences=[];self.labels=0;self.visual=0;self.loss_sum=0.;self.loss_count=0
  dist.barrier();torch.cuda.synchronize();t=time.perf_counter();cpu=time.process_time()
  self.step(iterator);self.clock.step_index+=1;self.metrics.clear();torch.cuda.synchronize();dist.barrier()
  duration=time.perf_counter()-t
  phases={}
  for name,a,b in self.events:phases[name]=phases.get(name,0)+a.elapsed_time(b)/1000
  raw=getattr(self.dataloader,'inner',self.dataloader)
  item={'rank':dist.get_rank(),'update':index,'seconds':duration,'cpu_seconds':time.process_time()-cpu,'data_wait_seconds':self.wait_seconds,'gpu_phase_seconds':phases,'loss':self.loss_sum/self.loss_count,'input_tokens':sum(self.sequences),'label_tokens':self.labels,'visual_tokens_estimate':self.visual,'sequence_lengths':self.sequences,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'read_seconds_cumulative':raw.read_seconds,'encode_seconds_cumulative':raw.encode_seconds,'kernel_calls':dict(CALLS)}
  gathered=[None]*4;dist.all_gather_object(gathered,item)
  if dist.get_rank()==0:
   with (OUT/'updates.jsonl').open('a') as f:f.write(json.dumps(gathered)+'\n')
   print('BENCH_UPDATE',MODE,index,'examples_s',256/max(x['seconds'] for x in gathered),'seconds',duration,flush=True)
  return item
 def _train(self):
  try:self.run_benchmark()
  except BaseException as exc:
   import traceback;traceback.print_exc();(OUT/f'failure-rank{dist.get_rank()}.txt').write_text(repr(exc));os._exit(1)
 def run_benchmark(self):
  rank=dist.get_rank();path=SOURCE/'pilot-10h-v1/checkpoint-3205.pt';sha='d9a45e38c05720708aaf24282f93050920a158aae8987aa449395c04d9597117'
  with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
  if rank==0:
   body=connect().get_object(Bucket=BUCKET,Key='pilot-10h-v1/checkpoint-3205.pt')['Body'];h=hashlib.sha256()
   for c in body.iter_chunks(8388608):h.update(c)
   body.close();assert h.hexdigest()==sha
  dist.barrier();DistributedTraining.restore(self,path)
  self.dataloader.load_state_dict({'cursor':0,'count':0,'pilot':True});self.config.gradient_accumulation.count=64
  iterator=iter(self.dataloader);initial=self.adapter_state()
  for index in range(4):self.run_update(index,iterator)
  assert any(not torch.equal(initial[k],v) for k,v in self.adapter_state().items());del initial
  # Flush the bounded producer before snapshotting the consumed state.
  if hasattr(iterator,'close'):iterator.close()
  snapshot=OUT/f'resume-rank{rank}.pt';OCRTrainer.save(self,snapshot)
  state=self.dataloader.state_dict();self.dataloader.load_state_dict(state);iterator=iter(self.dataloader)
  self.run_update(4,iterator);expected=self.adapter_state();loss=self.last_loss
  if hasattr(iterator,'close'):iterator.close()
  OCRTrainer.restore(self,snapshot);iterator=iter(self.dataloader)
  self.run_update(5,iterator)
  error=max(float((expected[k]-v).abs().max()) for k,v in self.adapter_state().items())
  assert error<1e-5 and abs(loss-self.last_loss)<1e-4,('Resume mismatch',error,loss,self.last_loss)
  if hasattr(iterator,'close'):iterator.close()
  self.sync_check()
  if MODE!='baseline':assert CALLS.get('chunk_gated_delta_rule',0)>0 and CALLS.get('causal_conv1d_fn',0)>0,'No optimized kernel executions'
  proofs=[None]*4;dist.all_gather_object(proofs,{'rank':rank,'resume_error':error,'kernel_calls':dict(CALLS)})
  if rank==0:
   records=[json.loads(x) for x in (OUT/'updates.jsonl').read_text().splitlines()]
   durations=[max(r['seconds'] for r in rs) for rs in records[1:4]]
   result={'status':'passed','mode':MODE,'steady_examples_per_second':768/sum(durations),'steady_seconds':sum(durations),'ranks':proofs,'source_checkpoint_sha256':sha,'sample_manifest_sha256':hashlib.sha256((SOURCE/'pilot-10h-v1/train.jsonl').read_bytes()).hexdigest(),'timed_steady_examples':768,'sample_limitations':'first1280 pilot train examples; no full-dataset throughput claim','torch':torch.__version__}
   (OUT/'result.json').write_text(json.dumps(result,indent=2));s=connect()
   for p in OUT.glob('*.json*'):
    key='optimization/'+OUT.name+'/'+p.name;s.upload_file(str(p),BUCKET,key);assert s.get_object(Bucket=BUCKET,Key=key)['Body'].read()==p.read_bytes()
   print('BENCH_COMPLETE',json.dumps(result),flush=True)
  dist.barrier();self.clock.stop()

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'updates.jsonl').exists(),'Do not overwrite benchmark'
 stop=threading.Event()
 def telemetry():
  while not stop.is_set():
   p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
   with (OUT/'gpu-telemetry.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'csv':p.stdout})+'\n')
   stop.wait(5)
 threading.Thread(target=telemetry,daemon=True).start()
 config=BenchConfig(run_directory=OUT/'arsenal',model=OCRModel(model_path=str(SOURCE/'model')),data=BenchData(model_path=str(SOURCE/'model'),batch_size=1),optimizer=AdamConfig(),lr_scheduler=CosineSchedulerConfig(init_lr=1e-4,max_lr=1e-4,min_lr_ratio=.1,max_steps=100000,use_warmup=False,warmup_steps=0,alpha=1.),distributed=DistributedConfig(device_type='cuda',num_process_per_node=4,hsdp=HsdpConfig(replicate_size=4,shard_size=1),use_local_loss=True),compile=CompileConfig(enabled=False),dtype=torch.bfloat16)
 try:BenchTrainer(config).train()
 finally:stop.set()
if __name__=='__main__':main()
