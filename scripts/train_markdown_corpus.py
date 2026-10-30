"""Single- or four-GPU DDP complete original+aug1 pass from adapter 14355.

Fresh optimizer/cursor for corrected targets, identical parent LoRA computation.
No old file is modified; no pruning or automatic retry of failed training.
"""
import argparse,contextlib,copy,datetime,hashlib,json,math,os,random,shutil,subprocess,threading,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from markdown_data_runtime import BUCKET,SOURCE_SHA,PREFIX,MAX_SEQUENCE,s3_client,read,retry,encode,atomic_json
from markdown_table_targets import sha,TARGET_FORMAT,VERSION,PROMPT
from markdown_stream import MarkdownStream,canonical_cursor
from table_correction_runtime import load_parent,PARENT_SHA,PARENT_STEP,digest
from table_phase_state import phase_lr
from ordered_prefetch import OrderedPrefetch
from minio_retry import read_object,object_sha256

WORLD=4;GLOBAL_BATCH=384

def main():
    global WORLD
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model-dir',required=True)
    p.add_argument('--world-size',type=int,choices=[1,4],default=4)
    p.add_argument('--resume',type=Path);a=p.parse_args();root=a.root;data=root/'data';output_prefix='training-markdown-v3-orig-aug1/'
    WORLD=a.world_size
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    import triton
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=WORLD:raise ValueError('Launcher rank count differs from requested world size')
    # Container devices are exclusively assigned by the scheduler; check before
    # creating any CUDA context. Rank zero checks every assigned device once.
    if rank==0:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,name','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        if len(visible)!=WORLD or any(int(line.split(',')[1])>256 or 'H200' not in line for line in visible):
            raise ValueError(f'Assigned GPUs must be {WORLD} unused H200 devices')
        for line in visible:
            uuid=line.split(',')[0].strip()
            if subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise ValueError('Assigned GPU has an existing compute process')
    # This file rendezvous avoids an early CUDA context racing rank0's check.
    availability=root/'gpu-availability.json';execution_id=os.environ['POD_UID']
    if rank==0:atomic_json(availability,{'checked':True,'pid':os.getpid(),'timestamp':time.time(),'execution_id':execution_id})
    else:
        began=time.time()
        while not availability.exists() or json.loads(availability.read_text()).get('execution_id')!=execution_id:
            if time.time()-began>120:raise ValueError('GPU availability check did not finish')
            time.sleep(.1)
    if a.resume is None and (root/'started.json').exists():raise ValueError('Run already started; explicit verified --resume required')
    if triton.__version__!='3.7.1' or not qwen.is_fast_path_available:raise ValueError('Verified Hopper kernels unavailable')
    torch.set_num_threads(1);torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30));torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed(42)
    calls={}
    for name in ['causal_conv1d_fn','chunk_gated_delta_rule']:
        fn=getattr(qwen,name)
        def counted(*args,_fn=fn,_name=name,**kwargs):
            calls[_name]=calls.get(_name,0)+1;return _fn(*args,**kwargs)
        setattr(qwen,name,counted)
    s3=s3_client();staged=json.loads((data/'STAGED.json').read_text());raw=(data/'COMPLETE.json').read_bytes();complete=json.loads(raw)
    if not staged['passed'] or not complete['training_ready'] or complete['schema']!=VERSION or complete['target_format']!=TARGET_FORMAT:
        raise ValueError('Whole-corpus preparation gates have not passed')
    data_sha=sha(raw)
    if data_sha!=staged['corpus_complete_sha256'] or read(s3,PREFIX+'COMPLETE.json')!=raw or complete['source_complete_sha256']!=SOURCE_SHA:
        raise ValueError('Dataset provenance differs')
    if staged['parent_sha256']!=PARENT_SHA or staged['train_examples']!=1841540:raise ValueError('Parent/corpus changed')
    if shutil.disk_usage(root).free<10*1024**3:raise ValueError('Disk reserve insufficient; no files deleted')
    for sub in ['checkpoints','logs','cache','ocr-before','ocr-after']:(root/sub).mkdir(parents=True,exist_ok=True)
    def on_rank0(operation):
        box=[None]
        if rank==0:
            try:box[0]={'ok':True,'value':operation()}
            except Exception as exc:box[0]={'ok':False,'error_type':type(exc).__name__,'error':str(exc)}
        dist.broadcast_object_list(box,src=0)
        if not box[0]['ok']:raise RuntimeError(box[0]['error_type']+': '+box[0]['error'])
        return box[0]['value']
    def upload(path):
        key=output_prefix+path.relative_to(root).as_posix();checksum=digest(path)
        def send():
            s3.upload_file(str(path),BUCKET,key,ExtraArgs={'Metadata':{'sha256':checksum}})
            if object_sha256(s3,BUCKET,key)!=checksum:raise ValueError('Training artifact SHA readback differs')
        retry(send);return {'path':str(path),'key':key,'sha256':checksum}
    def publish_json(name,value):
        def operation():
            path=root/name;atomic_json(path,value);return upload(path)
        return on_rank0(operation)
    on_rank0(lambda: object_sha256(s3,BUCKET,staged['parent_s3_key'])==PARENT_SHA or (_ for _ in ()).throw(ValueError('S3 parent changed')))
    base=load_parent(a.model_dir,Path(staged['parent_checkpoint']),device)
    model=DDP(base,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False)
    params=[p for p in base.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=29097984:raise ValueError('LoRA target parameters differ')
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    validation_processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    total=staged['train_examples'];steps=math.ceil(total/GLOBAL_BATCH)
    config={'phase':'full-markdown-correction','target_format':TARGET_FORMAT,'parent_step':PARENT_STEP,'parent_sha256':PARENT_SHA,
            'phase_data_sha256':data_sha,'source_data_sha256':SOURCE_SHA,'train_examples':total,'epochs':1,
            'gpus':WORLD,'strategy':'DDP','microbatch':1,'accumulation':GLOBAL_BATCH//WORLD,'global_batch':GLOBAL_BATCH,
            'maximum_sequence':MAX_SEQUENCE,'image_thumbnail_limit':1024,'lora_rank':16,'lora_alpha':32,'lora_compute':'fp32',
            'trainable_parameters':29097984,'base_dtype':'bf16','optimizer':'AdamW','weight_decay':.01,
            'gradient_clip':1.,'maximum_lr':2e-5,'minimum_lr':1e-5,'warmup_updates':100,'seed':42,
            'coverage':'all accepted original train pages plus their aug1; validation/test excluded',
            'optimizer_state':'fresh for complete Markdown supervision; parent state untouched',
            'checkpoint_every_updates':250,'checkpoint_every_seconds':3600,'validation_tables':128,'validation_control':128,
            'microbatch_policy':'one page for new longer targets; global batch preserved by accumulation',
            'code_sha256':{n:sha(Path(__file__).with_name(n).read_bytes()) for n in
                           ['train_markdown_corpus.py','markdown_stream.py','markdown_data_runtime.py','markdown_table_targets.py','lora.py']}}
    config_sha=sha(json.dumps(config,sort_keys=True));publish_json('config.json',config)
    optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
    stream=MarkdownStream(complete['shards'],s3,processor,rank,WORLD);loader=OrderedPrefetch(stream,depth=4);iterator=None
    phase_step=0;phase_samples=0;history=[];health={'best_tables':None,'best_overall':None,'control_baseline':None,'degraded':0}
    def close_loader():
        nonlocal iterator
        cursor=loader.state_dict();loader.stop_event.set()
        if loader.worker is not None:loader.worker.join(timeout=450)
        loader.load_state_dict(cursor);iterator=None
    def adapter():return {'hf.'+n:p.detach().cpu().clone() for n,p in base.named_parameters() if p.requires_grad}
    def sync_weights():
        digestor=hashlib.sha256()
        for name,p in adapter().items():digestor.update(name.encode());digestor.update(p.numpy().tobytes())
        hashes=[None]*WORLD;dist.all_gather_object(hashes,digestor.hexdigest())
        if len(set(hashes))!=1:raise ValueError('DDP adapter parameters diverged')
    def save(label,publish=True):
        close_loader();states=[None]*WORLD;dist.all_gather_object(states,loader.state_dict());cursor=canonical_cursor(states,phase_samples)
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        rngs=[None]*WORLD;dist.all_gather_object(rngs,rng);sync_weights()
        def operation():
            path=root/'checkpoints'/('checkpoint-'+label+'.pt')
            if path.exists():raise ValueError('Never overwrite a phase checkpoint')
            state={'adapter':adapter(),'optimizer':copy.deepcopy(optimizer.state_dict()),'clock':{'step_index':PARENT_STEP+phase_step},
                   'loader':cursor,'rank_rng':rngs,'phase_step':phase_step,'phase_samples':phase_samples,
                   'data_complete_sha256':data_sha,'source_data_complete_sha256':SOURCE_SHA,'phase_config_sha256':config_sha,
                   'phase_config':config,'source_checkpoint_sha256':PARENT_SHA,'validation_health':health.copy(),'loss_history':history[-20:]}
            tmp=path.with_suffix('.partial');torch.save(state,tmp);tmp.replace(path)
            marker=upload(path) if publish else {'path':str(path),'sha256':digest(path)}
            return {**marker,'phase_step':phase_step,'phase_samples':phase_samples,'global_step':PARENT_STEP+phase_step}
        return on_rank0(operation)
    def restore(marker):
        nonlocal phase_step,phase_samples
        close_loader();path=Path(marker['path'])
        if digest(path)!=marker['sha256']:raise ValueError('Resume checkpoint digest differs')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['phase_config_sha256']!=config_sha or state['data_complete_sha256']!=data_sha:raise ValueError('Exact resume configuration/data differ')
        with torch.no_grad():
            for name,p in base.named_parameters():
                if p.requires_grad:p.copy_(state['adapter']['hf.'+name])
        optimizer.load_state_dict(state['optimizer']);loader.load_state_dict(state['loader'])
        phase_step=state['phase_step'];phase_samples=state['phase_samples']
        if state['clock']['step_index']!=PARENT_STEP+phase_step or loader.count!=phase_samples:raise ValueError('Resume cursor/step differs')
        health.clear();health.update(state['validation_health']);history.clear();history.extend(state['loss_history'])
        rng=state['rank_rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        optimizer.zero_grad(set_to_none=True);model.train();sync_weights()
    def advance(global_count):
        nonlocal iterator,phase_step,phase_samples
        if global_count%WORLD or global_count<1 or phase_samples+global_count>total:raise ValueError('Invalid DDP update coverage')
        count=global_count//WORLD
        if iterator is None:iterator=iter(loader)
        # Complete each rank's reads before its first forward. One readiness
        # collective coordinates network errors without per-microstep barriers.
        batches=[];error=None;data_started=time.time()
        try:
            for _ in range(count):batches.append(next(iterator))
        except Exception as exc:error={'rank':rank,'error_type':type(exc).__name__,'error':str(exc)}
        readiness=[None]*WORLD;dist.all_gather_object(readiness,error)
        if any(readiness):raise RuntimeError('Data preparation failed on a rank: '+json.dumps([e for e in readiness if e]))
        data_seconds=time.time()-data_started;model.train();optimizer.zero_grad(set_to_none=True)
        rate=phase_lr(phase_step,steps,maximum=2e-5,minimum=1e-5,warmup=100)
        for group in optimizer.param_groups:group['lr']=rate
        loss_sum=torch.zeros((),device=device,dtype=torch.float64);bad=torch.zeros((),device=device,dtype=torch.int32)
        tokens=sum(int((b['labels'][:,1:]!=-100).sum()) for b in batches);before=dict(calls)
        torch.cuda.synchronize();started=time.time()
        for index,batch in enumerate(batches):
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            context=model.no_sync() if index<count-1 else contextlib.nullcontext()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                bad+=~torch.isfinite(loss);loss_sum+=loss.detach().double();(loss/count).backward()
            del batch,loss
        dist.all_reduce(bad)
        if int(bad):raise FloatingPointError('Nonfinite loss; optimizer not updated')
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=False)
        finite=(torch.isfinite(norm)&(norm>0)).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not int(finite):raise FloatingPointError('Nonfinite/zero DDP gradients; optimizer not updated')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        finite=torch.stack([torch.isfinite(p).all() for p in params]).all().int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not int(finite):raise FloatingPointError('Nonfinite DDP parameter')
        dist.all_reduce(loss_sum);token_sum=torch.tensor(tokens,device=device);dist.all_reduce(token_sum)
        phase_step+=1;phase_samples+=global_count;torch.cuda.synchronize()
        kernel_calls={n:calls.get(n,0)-before.get(n,0) for n in ['causal_conv1d_fn','chunk_gated_delta_rule']}
        if not all(kernel_calls.values()):raise ValueError('Actual update did not execute required fast kernels')
        measurement={'rank':rank,'data_seconds':data_seconds,'compute_seconds':time.time()-started,
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                     'kernel_calls':kernel_calls};measurements=[None]*WORLD;dist.all_gather_object(measurements,measurement)
        return {'loss':float(loss_sum)/global_count,'examples':global_count,'supervised_tokens':int(token_sum),
                'seconds':time.time()-started,'data_seconds':data_seconds,'lr':rate,'gradient_norm':float(norm),'kernel_calls':kernel_calls,'ranks':measurements}
    monitoring={}
    for category,m in staged['monitoring'].items():
        path=Path(m['path'])
        if not path.is_relative_to(data) or digest(path)!=m['sha256']:raise ValueError('Monitoring set changed')
        monitoring[category]=[json.loads(line) for line in path.read_text().splitlines()]
        if len(monitoring[category])!=m['samples'] or any(r['split'] not in ['validation','test'] for r in monitoring[category]):raise ValueError('Monitoring coverage differs')
    validation_cache={}
    def image_bytes(row):
        im=row['image'];start=im['offset'];return read_object(s3,BUCKET,im['key'],byte_range=f"bytes={start}-{start+im['size']-1}")
    def evaluate():
        close_loader();base.eval();values={}
        with torch.no_grad():
            for category in ['validation-tables','validation-control']:
                stats=torch.zeros(3,device=device,dtype=torch.float64)
                for row in monitoring[category][rank::WORLD]:
                    if row['id'] not in validation_cache:validation_cache[row['id']]=encode(validation_processor,row,image_bytes(row),pin=True)
                    source=validation_cache[row['id']];tokens=int((source['labels'][:,1:]!=-100).sum());batch={k:v.to(device,non_blocking=True) for k,v in source.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=base(**batch,use_cache=False).loss
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite validation loss')
                    stats+=torch.tensor([float(loss)*tokens,tokens,1],device=device,dtype=torch.float64);del batch,loss
                dist.all_reduce(stats);values[category]={'loss':float(stats[0]/stats[1]),'supervised_tokens':int(stats[1]),'samples':int(stats[2])}
        base.train();result={'phase_step':phase_step,'phase_samples':phase_samples,'values':values,
                            'overall_loss':sum(v['loss']*v['supervised_tokens'] for v in values.values())/sum(v['supervised_tokens'] for v in values.values())}
        def record():
            with (root/'evals.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            upload(root/'evals.jsonl');print('MARKDOWN_TRAIN_EVAL',json.dumps(result),flush=True)
        on_rank0(record);return result
    def preflight():
        if total-phase_samples<32:raise ValueError('Too few remaining samples for exact recovery gate')
        start_step=phase_step;start_samples=phase_samples
        initial=save(f'initial-{start_step:06d}',publish=False);original=adapter()
        first=advance(16);changed=any(not torch.equal(original[k],v) for k,v in adapter().items());del original
        if not changed:raise ValueError('DDP adapter did not update')
        checkpoint=save(f'preflight-{start_step:06d}');expected_loss=advance(16)['loss'];expected=adapter()
        states=[None]*WORLD;dist.all_gather_object(states,loader.state_dict());expected_cursor=canonical_cursor(states,phase_samples)
        restore(checkpoint);repeated_loss=advance(16)['loss'];error=max(float((expected[k]-v).abs().max()) for k,v in adapter().items());del expected
        states=[None]*WORLD;dist.all_gather_object(states,loader.state_dict())
        if canonical_cursor(states,phase_samples)!=expected_cursor:raise ValueError('Exact replay changed data cursor')
        if error>=1e-5 or abs(expected_loss-repeated_loss)>=1e-5:raise ValueError('Exact DDP checkpoint recovery failed')
        proof={'rank':rank,'parameter_max_error':error,'loss_error':abs(expected_loss-repeated_loss),'kernel_calls':dict(calls),'gpu':torch.cuda.get_device_name()}
        proofs=[None]*WORLD;dist.all_gather_object(proofs,proof);restore(initial)
        if phase_samples!=start_samples or phase_step!=start_step or loader.count!=start_samples:raise ValueError('Preflight not rolled back')
        result={'passed':True,'ranks':proofs,'verified_pvc_s3_checkpoint':checkpoint,'rolled_back_before_training':True,
                'gates':['complete-corpus-markdown','finite-gradients','parameter-update','DDP-equal-weights','exact-resume','actual-FLA-conv','new-cursor-zero']}
        publish_json('preflight.json' if a.resume is None else f'preflight-resume-{start_step:06d}.json',result)
        if rank==0:print('MARKDOWN_TRAIN_PREFLIGHT_PASSED',json.dumps(result),flush=True)
    def diagnostic(label):
        import io
        from PIL import Image
        from markdown_ocr_metrics import score
        close_loader();base.eval();results=[]
        # Same genuinely held-out originals before and after; no target content
        # is present in the generation input. Raw predictions remain unchanged.
        selected=monitoring['test-tables'][:8]+monitoring['test-control'][:4]
        eos=list(dict.fromkeys(x for x in [validation_processor.tokenizer.eos_token_id,
            validation_processor.tokenizer.convert_tokens_to_ids('<|im_end|>')] if isinstance(x,int) and x>=0))
        if not eos:raise ValueError('Missing EOS termination policy')
        with torch.no_grad():
            for row in selected[rank::WORLD]:
                with Image.open(io.BytesIO(image_bytes(row))) as im:image=im.convert('RGB')
                image.thumbnail((1024,1024))
                messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
                batch=validation_processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                    return_dict=True,return_tensors='pt',enable_thinking=False)
                batch={k:v.to(device) for k,v in batch.items()};started=time.time()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=base.generate(**batch,do_sample=False,use_cache=True,max_new_tokens=6000,max_time=300,
                                         eos_token_id=eos,pad_token_id=eos[0])
                tokens=output[0,batch['input_ids'].shape[1]:].tolist()
                prediction=validation_processor.tokenizer.decode(tokens,skip_special_tokens=True)
                item={'id':row['id'],'group_id':row['group_id'],'split':row['split'],'phase_step':phase_step,'input_contains_ground_truth':False,
                      'new_tokens':len(tokens),'ended_with_eos':bool(tokens and tokens[-1] in eos),'seconds':time.time()-started,
                      'reference_table_count':row['table_count'],'reference_source_cells':row['source_cell_count'],
                      'metrics':score(row['text'],prediction),'prediction_sha256':sha(prediction)}
                filename=row['id'].replace('/','__').replace(':','__')
                path=root/label/(filename+'.md');path.write_text(prediction,encoding='utf-8');upload(path)
                path=root/label/(filename+'.json');atomic_json(path,item);upload(path);results.append(item)
                del batch,output
        gathered=[None]*WORLD;dist.all_gather_object(gathered,results);base.train()
        summary={'phase_step':phase_step,'samples':len(selected),'results':[r for part in gathered for r in part],
                 'same_held_out_originals_before_after':True,'prediction_links_never_filled_with_ground_truth':True,
                 'inference_output_limit':6000,'training_context_unchanged':MAX_SEQUENCE}
        publish_json(label+'/summary.json',summary)
        if rank==0:print('MARKDOWN_OCR_DIAGNOSTIC',json.dumps({'label':label,'samples':len(selected),'phase_step':phase_step}),flush=True)
        return summary
    telemetry_stop=threading.Event();telemetry_lock=threading.Lock()
    def telemetry():
        while not telemetry_stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
                with telemetry_lock,(root/'gpu-telemetry.jsonl').open('a') as f:
                    for line in raw.splitlines():
                        v=[s.strip() for s in line.split(',')];f.write(json.dumps({'timestamp':time.time(),'uuid':v[0],'utilization_percent':float(v[1]),'memory_used_mib':float(v[2]),'memory_total_mib':float(v[3]),'power_watts':float(v[4])})+'\n')
            except Exception:pass
            telemetry_stop.wait(5)
    thread=threading.Thread(target=telemetry,daemon=True) if rank==0 else None
    if thread:thread.start()
    try:
        if a.resume is not None:
            marker=json.loads((root/'latest.json').read_text())
            if Path(marker['path'])!=a.resume:raise ValueError('Resume must name latest verified phase checkpoint')
            on_rank0(lambda: object_sha256(s3,BUCKET,marker['key'])==marker['sha256'] or (_ for _ in ()).throw(ValueError('Resume S3 digest differs')))
            restore(marker)
        preflight()
        if a.resume is None:
            baseline=evaluate();health.update(best_tables=baseline['values']['validation-tables']['loss'],best_overall=baseline['overall_loss'],control_baseline=baseline['values']['validation-control']['loss'])
            marker=save('000000');publish_json('latest.json',marker);publish_json('best.json',{**marker,'eval_loss':health['best_overall']});publish_json('baseline.json',baseline)
            publish_json('started.json',{'timestamp':time.time(),'preflight_passed':True,'initial_verified_checkpoint':marker,'config':config})
            diagnostic('ocr-before')
        else:baseline=json.loads((root/'baseline.json').read_text())
        began=time.time();initial_samples=phase_samples;last_save=began;stop_reason=None
        while phase_samples<total:
            reason=on_rank0(lambda:'requested_stop' if (root/'STOP').exists() else 'disk_reserve_reached_no_files_deleted' if shutil.disk_usage(root).free<4*1024**3 else None)
            if reason:stop_reason=reason;break
            update_started=time.time();update=advance(min(GLOBAL_BATCH,total-phase_samples));history.append(update['loss']);history[:]=history[-20:]
            if update['loss']>max(20.,baseline['overall_loss']*10):raise FloatingPointError('Unstable full Markdown training loss')
            if len(history)>=13 and all(x>3*np.median(history[-13:-3]) for x in history[-3:]):stop_reason='divergent_loss'
            metric={'timestamp':time.time(),'state':'running','global_step':PARENT_STEP+phase_step,'phase_step':phase_step,
                    'phase_samples':phase_samples,'total_phase_samples':total,'train_loss':update['loss'],'learning_rate':update['lr'],
                    'update_seconds':time.time()-update_started,'data_seconds':update['data_seconds'],'compute_seconds':update['seconds'],
                    'world_size':WORLD,'global_batch':GLOBAL_BATCH,'microbatch':1,'accumulation':update['examples']//WORLD,
                    'supervised_tokens':update['supervised_tokens'],'gradient_norm':update['gradient_norm'],
                    'rank_measurements':update['ranks'],
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                    'examples_per_second':(phase_samples-initial_samples)/(time.time()-began),'actual_kernel_calls':dict(calls)}
            save_now=phase_step==1 or phase_step%250==0 or phase_samples==total or time.time()-last_save>=3600 or bool(stop_reason)
            if save_now:
                score=evaluate();metric['eval_loss']=score['overall_loss'];table_loss=score['values']['validation-tables']['loss'];control_loss=score['values']['validation-control']['loss']
                health['degraded']=health['degraded']+1 if table_loss>health['best_tables']*1.2 and table_loss-health['best_tables']>.05 or control_loss>health['control_baseline']*1.3 else 0
                if health['degraded']>=2:stop_reason='repeated_validation_degradation'
                improved=score['overall_loss']<health['best_overall'];health['best_tables']=min(health['best_tables'],table_loss);health['best_overall']=min(health['best_overall'],score['overall_loss'])
                marker=save(f'{phase_step:06d}');publish_json('latest.json',marker)
                if improved:publish_json('best.json',{**marker,'eval_loss':score['overall_loss']})
                def backup_telemetry():
                    if (root/'gpu-telemetry.jsonl').exists():
                        path=root/'logs'/f'gpu-telemetry-step{phase_step:06d}.jsonl'
                        with telemetry_lock:shutil.copyfile(root/'gpu-telemetry.jsonl',path)
                        upload(path)
                on_rank0(backup_telemetry)
                last_save=time.time()
            def record():
                with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                atomic_json(root/'status.json',metric)
                if save_now or phase_step%10==0:upload(root/'metrics.jsonl');upload(root/'status.json')
                print('MARKDOWN_TRAIN_METRIC',json.dumps(metric),flush=True)
            on_rank0(record)
            if stop_reason:break
        final_eval=evaluate()
        if stop_reason:
            marker=save(f'stopped-{phase_step:06d}');publish_json('latest.json',marker)
        else:marker=json.loads((root/'latest.json').read_text())
        diagnostic('ocr-after' if stop_reason is None else 'ocr-stopped')
        result={'state':'stopped' if stop_reason else 'complete','stop_reason':stop_reason,'phase_step':phase_step,'phase_samples':phase_samples,
                'total_phase_samples':total,'latest_verified_checkpoint':marker,'final_validation':final_eval,'baseline_validation':baseline,
                'parent_full_pass_preserved':True,'tables_fixed_claim':False,
                'held_out_ocr_before':'ocr-before/summary.json','held_out_ocr_after':'ocr-after/summary.json' if stop_reason is None else 'ocr-stopped/summary.json',
                'next_gate':'review real saved held-out OCR, cell completeness and normalized CER/WER; no automatic claim of a universal fix'}
        publish_json('result.json',result)
        if rank==0:print('MARKDOWN_TRAIN_RESULT',json.dumps(result),flush=True)
    except Exception as exc:
        atomic_json(root/f'failure-rank{rank}.json',{'state':'failed','error_type':type(exc).__name__,'error':str(exc),'phase_samples':phase_samples,'phase_step':phase_step})
        raise
    finally:
        telemetry_stop.set()
        if thread:thread.join(timeout=6)
        close_loader();dist.destroy_process_group()

if __name__=='__main__':main()
