import json,time,datetime,hashlib
from pathlib import Path
root=Path('/work/q35ocr-20260921-p560/training-markdown-v3-orig-aug1')
expected_job_uid='9508ac42-8510-4397-9726-48c90774d45a'
expected_pod_uid=EXPECTED_POD_UID
deadline=time.monotonic()+24*3600
print('FINAL_VALIDATION_DEPENDENCY_WAIT: no GPU and no training modifications',flush=True)
while time.monotonic()<deadline:
 p=root/'resume-20261026-execution.json'
 execution=json.loads(p.read_text()) if p.exists() else {}
 if execution:
  assert execution['pod_uid']==expected_pod_uid,'Unexpected recovery execution'
 if execution.get('state') in ('failed','stopped'):raise RuntimeError('Training stopped before complete; no success verification')
 if execution.get('state')=='training_complete_pending_independent_verification':break
 time.sleep(60)
else:raise TimeoutError('Training completion dependency was not satisfied')
resume=json.loads((root/'preflight-resume-004682.json').read_text())
assert resume['passed'] and resume['rolled_back_before_training'] and len(resume['ranks'])==1
assert resume['ranks'][0]['rank']==0 and resume['ranks'][0]['parameter_max_error']<1e-5 and resume['ranks'][0]['loss_error']<1e-5
assert all(resume['ranks'][0]['kernel_calls'].get(k,0)>0 for k in ('causal_conv1d_fn','chunk_gated_delta_rule'))

import hashlib,json,os
from pathlib import Path
import boto3
from botocore.config import Config
from minio_retry import read_object,object_sha256
root=Path('/work/q35ocr-20260921-p560/training-markdown-v3-orig-aug1')
bucket='qwen35-9b-ocr-finetuning';prefix='training-markdown-v3-orig-aug1/'
s3=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'],region_name='us-east-1',
    config=Config(connect_timeout=10,read_timeout=45,retries={'max_attempts':0},s3={'addressing_style':'path'}))
result_path=root/'result.json';result=json.loads(result_path.read_text())
assert result['state']=='complete' and result['stop_reason'] is None
assert result['phase_samples']==result['total_phase_samples']==1841540
assert read_object(s3,bucket,prefix+'result.json')==result_path.read_bytes()
preflight=json.loads((root/'preflight.json').read_text())
assert preflight['passed'] and len(preflight['ranks'])==1 and preflight['ranks'][0]['rank']==0 and preflight['rolled_back_before_training']
assert all(r['parameter_max_error']<1e-5 and r['loss_error']<1e-5 for r in preflight['ranks'])
assert all(all(r['kernel_calls'].get(k,0)>0 for k in ['causal_conv1d_fn','chunk_gated_delta_rule']) for r in preflight['ranks'])
marker=result['latest_verified_checkpoint'];path=Path(marker['path'])
assert path.is_relative_to(root/'checkpoints') and marker['key'].startswith(prefix+'checkpoints/')
with path.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
assert checksum==marker['sha256']==object_sha256(s3,bucket,marker['key'])
assert marker['phase_samples']==1841540 and marker['phase_step']==result['phase_step']
import torch
state=torch.load(path,map_location='cpu',weights_only=False)
assert state['phase_samples']==1841540 and state['phase_config']['phase']=='full-markdown-correction'
assert state['phase_config']['gpus']==1 and state['phase_config']['global_batch']==384 and state['phase_config']['accumulation']==384
assert len(state['adapter'])==256 and all(torch.isfinite(t).all() for t in state['adapter'].values())
complete_path=root/'data/COMPLETE.json'
assert hashlib.sha256(complete_path.read_bytes()).hexdigest()==state['data_complete_sha256']
assert read_object(s3,bucket,'dataset/processed/ocr-markdown-v3-orig-aug1/COMPLETE.json')==complete_path.read_bytes()
complete=json.loads(complete_path.read_text());assert complete['training_ready'] and len(complete['shards'])==206
assert complete['counts']['train']==1841540 and complete['counts']['samples']==2038290
before=json.loads((root/'ocr-before/summary.json').read_text());after=json.loads((root/'ocr-after/summary.json').read_text())
assert before['samples']==after['samples']==12
assert {r['id'] for r in before['results']}=={r['id'] for r in after['results']}
assert all(r['split']=='test' and r['input_contains_ground_truth'] is False for r in before['results']+after['results'])
last=None
with (root/'metrics.jsonl').open() as f:
 for line in f:
  if line.strip():last=json.loads(line)
assert last and last['phase_samples']==1841540
assert last['world_size']==1
payload={'state':'verified_complete','phase_samples':1841540,'total_phase_samples':1841540,
         'phase_step':result['phase_step'],'global_step':marker['global_step'],'world_size':1,
         'train_loss':last['train_loss'],'eval_loss':result['final_validation']['overall_loss'],
         'eval_tables':result['final_validation']['values']['validation-tables'],
         'eval_control':result['final_validation']['values']['validation-control'],
         'checkpoint':marker,'pvc_s3_sha256_match':True,'all_rank_preflight_passed':True,
         'corrected_dataset_complete':True,'ocr_before_after_saved':True,
         'tables_fixed_universal_claim':False,'artifact_prefix':'s3://'+bucket+'/'+prefix}
assert state['loader']['count']==1841540 and state['clock']['step_index']==14355+result['phase_step']
assert state['phase_config']['maximum_sequence']==8192 and state['phase_config']['lora_rank']==16 and state['phase_config']['lora_alpha']==32
assert len(state['rank_rng'])==1 and len(state['optimizer']['state'])==256
assert hashlib.sha256(json.dumps(state['phase_config'],sort_keys=True).encode()).hexdigest()==state['phase_config_sha256']
for name,expected_sha in state['phase_config']['code_sha256'].items():
 assert hashlib.sha256((Path('/original-code')/name).read_bytes()).hexdigest()==expected_sha
assert execution['phase_samples']==1841540 and execution['phase_step']==result['phase_step']
assert execution['resume_phase_samples']==1797888 and execution['resume_phase_step']==4682
payload.update({'job_uid':expected_job_uid,'pod_uid':expected_pod_uid,'resume_preflight_passed':True,
 'exact_cursor_consumed':True,'original_training_sources_unchanged':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
Path('/dev/termination-log').write_text(json.dumps(payload))
print('TRAINING_COMPLETION_VERIFIED '+json.dumps(payload),flush=True)
