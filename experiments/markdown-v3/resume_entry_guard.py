import datetime,fcntl,hashlib,json,os,runpy,shutil,time
from pathlib import Path
root=Path('/work/q35ocr-20260921-p560/training-markdown-v3-orig-aug1')
lock=(root/'resume-20261026.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
marker=json.loads((root/'latest.json').read_text())
assert marker['phase_step']==4682 and marker['phase_samples']==1797888
assert marker['sha256']=='db14c58ed66ca6ffc4cbc140d03da9f317ec89166ea94dfbebab4472fae18cbe'
with Path(marker['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
assert not (root/'STOP').exists(),'STOP present: no training started'
assert shutil.disk_usage(root).free>20*1024**3,'Storage extension not yet visible: no training started'
# Create the missing diagnostic output directory without changing the exact
# hashed training sources or checkpoint configuration.
(root/'ocr-stopped').mkdir(exist_ok=True)
record=root/'resume-20261026-execution.json'
def write(state,**extra):
 value={'state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pod_uid':os.environ['POD_UID'],'resume_phase_step':4682,'resume_phase_samples':1797888,
        'checkpoint_sha256':marker['sha256'],'remaining_examples_at_resume':43652,
        'original_training_sources_unchanged':True,**extra}
 tmp=record.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(record)
write('starting_preflight_and_resume',disk_free_bytes=shutil.disk_usage(root).free)
print('MARKDOWN_RECOVERY_BEGIN',record.read_text(),flush=True)
try:
 runpy.run_path('/corpus-code/entry-original.py',run_name='__main__')
 result=json.loads((root/'result.json').read_text())
 complete=result['state']=='complete' and result['phase_samples']==result['total_phase_samples']==1841540 and result['stop_reason'] is None
 write('training_complete_pending_independent_verification' if complete else 'stopped',
       phase_samples=result['phase_samples'],phase_step=result['phase_step'],stop_reason=result['stop_reason'])
 if not complete:raise SystemExit(2)
except BaseException as error:
 if not isinstance(error,SystemExit):write('failed',error_type=type(error).__name__)
 raise
