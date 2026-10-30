"""One-GPU isolated tables-only correction from verified checkpoint 14355.

Only adapter weights are inherited. New objective, optimizer, schedule and cursor;
real update/replay/S3/kernel gates are rolled back before the corrective pass.
"""
import argparse,copy,hashlib,json,math,os,random,shutil,subprocess,threading,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from table_correction_runtime import PARENT_SHA,PARENT_STEP,SOURCE_DATA_SHA,MAX_SEQUENCE,digest,encode,load_parent
from table_phase_state import PhaseCursor,phase_lr
from ordered_prefetch import OrderedPrefetch
from minio_retry import read_object,object_sha256,retryable,RETRY_DELAYS

BUCKET='qwen35-9b-ocr-finetuning';BATCH=32

def write(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-dir',required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    root=args.root;data=root/'data';prefix='training-tables-correction-v1/'
    if (root/'started.json').exists() and args.resume is None:raise ValueError('Corrective run already started; explicit --resume required')
    complete=json.loads((data/'COMPLETE.json').read_text(encoding='utf-8'));data_sha=digest(data/'COMPLETE.json')
    assert complete['schema']=='table-correction-pilot-v1' and complete['source_data_complete_sha256']==SOURCE_DATA_SHA
    assert complete['all_data_files_s3_sha256_verified'] and complete['maximum_sequence']==MAX_SEQUENCE
    assert digest(Path(__file__).with_name('table_correction_runtime.py'))==complete['encoder_sha256']
    assert digest(Path(__file__).with_name('annotation_targets.py'))==complete['target_builder_sha256']
    # Check exclusive availability before initializing a CUDA model.
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,name','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(visible)==1,'Exactly one assigned GPU required'
    gpu=visible[0].split(',');assert int(gpu[1].strip())<=256 and 'H200' in gpu[2]
    uuid=gpu[0].strip()
    assert not subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Assigned GPU is busy'
    if shutil.disk_usage(root).free<6*1024**3:raise RuntimeError('Disk reserve too low; no files deleted')
    for directory in ['checkpoints','cache/torch-kernels','cache/triton','cache/cuda','cache/xdg','logs']:(root/directory).mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    import triton
    assert triton.__version__=='3.7.1' and qwen.is_fast_path_available,'Required verified kernels are missing'
    assert torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    calls={}
    for name in ['causal_conv1d_fn','chunk_gated_delta_rule']:
        original=getattr(qwen,name)
        def counted(*a,_fn=original,_name=name,**kw):
            calls[_name]=calls.get(_name,0)+1;return _fn(*a,**kw)
        setattr(qwen,name,counted)
    import boto3
    from botocore.config import Config
    s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',
                   config=Config(connect_timeout=10,read_timeout=30,retries={'max_attempts':0},s3={'addressing_style':'path'}))
    assert read_object(s3,BUCKET,prefix+'data/COMPLETE.json')==(data/'COMPLETE.json').read_bytes()
    rows={}
    for category,m in complete['manifests'].items():
        path=Path(m['path']);assert path.is_relative_to(data) and digest(path)==m['sha256']
        rows[category]=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows[category])==m['samples']
    assert len({r['group_id'] for r in rows['train']})==len(rows['train'])
    assert all(r['split']=='train' and not r['augmented'] and r['table_audit']['table_count']>0 for r in rows['train'])
    training_groups={r['group_id'] for r in rows['train']}
    assert not training_groups & {r['group_id'] for k,v in rows.items() if k!='train' for r in v}
    total=len(rows['train']);steps=math.ceil(total/BATCH);device=torch.device('cuda',0)
    config={'phase':'tables-only-correction','parent_step':PARENT_STEP,'parent_sha256':PARENT_SHA,
            'source_data_sha256':SOURCE_DATA_SHA,'phase_data_sha256':data_sha,'epochs':1,'train_examples':total,
            'validation_table_examples':len(rows['validation-tables']),'validation_control_examples':len(rows['validation-control']),
            'reserved_test_table_examples':len(rows['test-tables']),'gpus':1,'microbatch':1,'accumulation':BATCH,'global_batch':BATCH,
            'maximum_sequence':MAX_SEQUENCE,'image_thumbnail_limit':1024,'lora_rank':16,'lora_alpha':32,'lora_compute':'fp32',
            'base_dtype':'bf16','optimizer':'AdamW','weight_decay':.01,'gradient_clip':1.,'maximum_lr':2e-5,'minimum_lr':1e-5,'warmup_updates':8,
            'optimizer_state':'fresh for the changed objective; historical optimizer/cursor preserved in parent',
            'seed':42,'checkpoint_every_updates':32,'kernel_versions':{'triton':triton.__version__},
            'engine':'PyTorch/Transformers with the identical verified custom LoRA forward path'}
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write(root/'config.json',config)
    model=load_parent(args.model_dir,args.parent,device)
    params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==29097984
    optimizer=torch.optim.AdamW(params,lr=phase_lr(0,steps),weight_decay=.01)
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    validation_processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    class Stream:
        def __init__(self):self.position=PhaseCursor(total)
        def state_dict(self):return self.position.state_dict()
        def load_state_dict(self,state):self.position.load_state_dict(state)
        def __iter__(self):
            while self.position.cursor<total:
                index=self.position.order[self.position.cursor];row=rows['train'][index]
                batch=encode(processor,row,pin=True)
                assert batch['input_ids'].shape[1]==row['sequence_tokens']
                assert int((batch['labels'][:,1:]!=-100).sum())==row['supervised_tokens']
                self.position.next_index()
                yield batch
    loader=OrderedPrefetch(Stream(),depth=4);iterator=None;phase_step=0;phase_samples=0
    health={'best':None,'degraded':0,'control_baseline':None}
    def close_loader():
        nonlocal iterator
        state=loader.state_dict();loader.load_state_dict(state);iterator=None
    def adapter():return {'hf.'+n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    def snapshot():
        return {'adapter':adapter(),'optimizer':copy.deepcopy(optimizer.state_dict()),
                'clock':{'step_index':PARENT_STEP+phase_step},'phase_step':phase_step,'phase_samples':phase_samples,
                'loader':loader.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'data_complete_sha256':data_sha,'source_data_complete_sha256':SOURCE_DATA_SHA,
                'phase_config_sha256':config_sha,'phase_config':config,'source_checkpoint_sha256':PARENT_SHA,
                'validation_health':copy.deepcopy(health)}
    def restore(state):
        nonlocal phase_step,phase_samples,iterator
        close_loader();assert state['phase_config_sha256']==config_sha and state['data_complete_sha256']==data_sha
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(state['adapter']['hf.'+name])
        optimizer.load_state_dict(state['optimizer']);loader.load_state_dict(state['loader'])
        phase_step=state['phase_step'];phase_samples=state['phase_samples'];iterator=None
        health.clear();health.update(state['validation_health'])
        assert state['clock']['step_index']==PARENT_STEP+phase_step and loader.state_dict()['cursor']==phase_samples
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        optimizer.zero_grad(set_to_none=True);model.train()
    def advance(requested):
        nonlocal iterator,phase_step,phase_samples
        count=min(requested,total-phase_samples);assert count>0
        if iterator is None:iterator=iter(loader)
        model.train();optimizer.zero_grad(set_to_none=True)
        rate=phase_lr(phase_step,steps)
        for group in optimizer.param_groups:group['lr']=rate
        losses=[];tokens=0;before_calls=dict(calls);torch.cuda.synchronize();started=time.time()
        for _ in range(count):
            batch=next(iterator);tokens+=int((batch['labels'][:,1:]!=-100).sum())
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite corrective loss')
            losses.append(float(loss.detach()));(loss/count).backward();del batch,loss
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if float(norm)<=0:raise FloatingPointError('Zero corrective gradient')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite corrective parameter')
        phase_step+=1;phase_samples+=count
        assert loader.state_dict()['cursor']==phase_samples
        torch.cuda.synchronize()
        update_calls={n:calls.get(n,0)-before_calls.get(n,0) for n in ['causal_conv1d_fn','chunk_gated_delta_rule']}
        assert all(n>0 for n in update_calls.values()),'Fast kernels did not execute during this actual update'
        return {'loss':sum(losses)/count,'examples':count,'supervised_tokens':tokens,'seconds':time.time()-started,'gradient_norm':float(norm),'lr':rate,'update_kernel_calls':update_calls}
    def upload(path):
        key=prefix+path.relative_to(root).as_posix();sha=digest(path)
        for attempt in range(4):
            try:
                s3.upload_file(str(path),BUCKET,key,ExtraArgs={'Metadata':{'sha256':sha}})
                assert object_sha256(s3,BUCKET,key)==sha
                return {'path':str(path),'key':key,'sha256':sha,'phase_step':phase_step,'phase_samples':phase_samples,'global_step':PARENT_STEP+phase_step}
            except Exception as exc:
                if not retryable(exc) or attempt==3:raise
                time.sleep(RETRY_DELAYS[attempt])
    def save(label):
        close_loader();path=root/'checkpoints'/('checkpoint-'+label+'.pt')
        assert not path.exists(),'Never overwrite a corrective checkpoint'
        temporary=path.with_suffix('.partial');torch.save(snapshot(),temporary);temporary.replace(path)
        return upload(path)
    def publish_json(name,value):
        path=root/name;write(path,value);return upload(path)
    validation_cache={}
    def evaluate():
        close_loader();model.eval();values={}
        with torch.no_grad():
            for category in ['validation-tables','validation-control']:
                weighted=0.;tokens=0
                for row in rows[category]:
                    if row['id'] not in validation_cache:validation_cache[row['id']]=encode(validation_processor,row,pin=True)
                    source_batch=validation_cache[row['id']];count=int((source_batch['labels'][:,1:]!=-100).sum())
                    batch={k:v.to(device,non_blocking=True) for k,v in source_batch.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite validation loss')
                    weighted+=float(loss)*count;tokens+=count;del batch,loss
                values[category]={'loss':weighted/tokens,'samples':len(rows[category]),'supervised_tokens':tokens}
        model.train();result={'phase_step':phase_step,'phase_samples':phase_samples,'values':values}
        with (root/'evals.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('TABLE_CORRECTION_EVAL',json.dumps(result),flush=True);return result
    def preflight():
        close_loader();initial=snapshot();initial_weights=initial['adapter']
        original_step=phase_step;original_samples=phase_samples
        assert total-phase_samples>=8,'Insufficient remaining examples for a fresh resume gate'
        first=advance(4);updated=adapter()
        assert any(not torch.equal(initial_weights[k],updated[k]) for k in initial_weights),'No corrective parameter update'
        label='preflight' if args.resume is None else f'preflight-resume-{original_step:06d}'
        entry=save(label);second=advance(4);expected=adapter();expected_state=loader.state_dict()
        state=torch.load(entry['path'],map_location='cpu',weights_only=False);restore(state);replayed=advance(4)
        error=max(float((expected[k]-p).abs().max()) for k,p in adapter().items())
        loss_error=abs(second['loss']-replayed['loss'])
        assert error<1e-5 and loss_error<1e-5 and loader.state_dict()==expected_state
        assert all(calls.get(n,0)>0 for n in ['causal_conv1d_fn','chunk_gated_delta_rule'])
        result={'passed':True,'resume_parameter_max_error':error,'resume_loss_error':loss_error,
                'actual_kernel_calls':dict(calls),'verified_recovery_checkpoint':entry,
                'gates':['complete_table_targets','source_parent_sha','finite_loss_and_gradients','actual_parameter_update','exact_phase_resume','verified_pvc_s3_checkpoint','actual_fast_kernel_execution'],
                'rolled_back_before_training':True}
        restore(initial);assert phase_step==original_step and phase_samples==loader.state_dict()['cursor']==original_samples
        assert all(torch.equal(initial_weights[k],p) for k,p in adapter().items())
        calls.clear();publish_json('preflight.json' if args.resume is None else f'preflight-resume-{original_step:06d}.json',result)
        print('TABLE_CORRECTION_PREFLIGHT_PASSED',json.dumps(result),flush=True)
    telemetry_stop=threading.Event();telemetry_lock=threading.Lock()
    def telemetry():
        while not telemetry_stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
                values=[float(v.strip()) for v in raw.strip().split(',')]
                with telemetry_lock,(root/'gpu-telemetry.jsonl').open('a') as f:f.write(json.dumps({'timestamp':time.time(),'gpu_uuid':uuid,'utilization_percent':values[0],'memory_used_mib':values[1],'memory_total_mib':values[2],'power_watts':values[3]})+'\n')
            except Exception:pass
            telemetry_stop.wait(5)
    thread=threading.Thread(target=telemetry,daemon=True);thread.start()
    try:
        if args.resume is not None:
            marker=json.loads((root/'latest.json').read_text());assert Path(marker['path'])==args.resume
            assert digest(args.resume)==marker['sha256'] and object_sha256(s3,BUCKET,marker['key'])==marker['sha256']
            restore(torch.load(args.resume,map_location='cpu',weights_only=False))
            assert phase_samples<total,'Phase is already complete'
        preflight()
        if args.resume is None:
            baseline=evaluate();health.update(best=baseline['values']['validation-tables']['loss'],control_baseline=baseline['values']['validation-control']['loss'],degraded=0)
            first_checkpoint=save('000000');publish_json('latest.json',first_checkpoint);publish_json('baseline.json',baseline)
            publish_json('started.json',{'timestamp':time.time(),'parent_checkpoint':str(args.parent),'config':config,'preflight_passed':True,'initial_verified_checkpoint':first_checkpoint})
        else:
            baseline=json.loads((root/'baseline.json').read_text())
            publish_json(f'resume-{phase_step:06d}.json',{'timestamp':time.time(),'resumed_verified_checkpoint':marker,'phase_samples':phase_samples,'preflight_passed':True})
        calls.clear();started=time.time();initial_samples=phase_samples;stop_reason=None
        while phase_samples<total:
            if (root/'STOP').exists():stop_reason='user_stop_file';break
            if shutil.disk_usage(root).free<4*1024**3:stop_reason='disk_reserve_reached_no_files_deleted';break
            update=advance(BATCH)
            if update['loss']>max(20.,baseline['values']['validation-tables']['loss']*10):raise FloatingPointError('Unstable corrective loss')
            elapsed=time.time()-started
            metric={'timestamp':time.time(),'state':'running','phase_step':phase_step,'global_step':PARENT_STEP+phase_step,
                    'phase_samples':phase_samples,'total_phase_samples':total,'train_loss':update['loss'],'learning_rate':update['lr'],
                    'update_seconds':update['seconds'],'examples_per_second':(phase_samples-initial_samples)/elapsed,'elapsed_seconds':elapsed,
                    'gpu_count':1,'microbatch':1,'global_batch':BATCH,'accumulation':update['examples'],
                    'supervised_tokens':update['supervised_tokens'],'gradient_norm':update['gradient_norm'],
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'actual_kernel_calls':dict(calls),
                    'actual_update_kernel_calls':update['update_kernel_calls']}
            write(root/'status.json',metric)
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            print('TABLE_CORRECTION_METRIC',json.dumps(metric),flush=True)
            evaluation=None
            if phase_step%32==0 or phase_samples==total:
                evaluation=evaluate();table_loss=evaluation['values']['validation-tables']['loss'];control_loss=evaluation['values']['validation-control']['loss']
                health['degraded']=health['degraded']+1 if table_loss>health['best']*1.1 or control_loss>health['control_baseline']*1.3 else 0
                if health['degraded']>=2:stop_reason='repeated_validation_degradation'
                improved=table_loss<health['best']
                if improved:health['best']=table_loss
            else:improved=False
            if phase_step==1 or phase_step%32==0 or phase_samples==total or stop_reason:
                entry=save(f'{phase_step:06d}');publish_json('latest.json',entry);publish_json('status.json',metric)
                if improved:publish_json('best.json',{**entry,'validation_table_loss':health['best']})
                with telemetry_lock:
                    snapshot_path=root/'logs'/f'gpu-telemetry-step{phase_step:06d}.jsonl';shutil.copyfile(root/'gpu-telemetry.jsonl',snapshot_path)
                upload(snapshot_path)
                print('TABLE_CORRECTION_CHECKPOINT_VERIFIED',json.dumps(entry),flush=True)
            if stop_reason:break
        final_eval=evaluate()
        if stop_reason:
            entry=save(f'stopped-{phase_step:06d}');publish_json('latest.json',entry)
        else:entry=json.loads((root/'latest.json').read_text())
        result={'state':'stopped' if stop_reason else 'complete','stop_reason':stop_reason,'phase_samples':phase_samples,'total_phase_samples':total,
                'phase_step':phase_step,'source_checkpoint_step':PARENT_STEP,'latest_verified_checkpoint':entry,'final_validation':final_eval,
                'baseline_validation':baseline,'best_validation_table_loss':health['best'],'original_full_pass_preserved':True,
                'tables_fixed_claim':False,'next_gate':'OCR generation and cell completeness on reserved held-out pages.'}
        publish_json('result.json',result);print('TABLE_CORRECTION_RESULT',json.dumps(result),flush=True)
    except Exception as exc:
        write(root/'failure.json',{'state':'failed','error_type':type(exc).__name__,'phase_step':phase_step,'phase_samples':phase_samples,'last_verified_checkpoint_exists':(root/'latest.json').exists()})
        raise
    finally:telemetry_stop.set();thread.join(timeout=6);close_loader()

if __name__=='__main__':main()
