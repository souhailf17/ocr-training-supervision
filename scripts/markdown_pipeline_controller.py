"""One-shot dependency orchestration: conversion -> final gate -> submit DDP.

No training monitoring/heartbeat, no workload deletion, no failed-job restart.
Runs on the authorized VM so laptop/VPN disconnection cannot break the pipeline.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path

NS='mtrna-ns1';PROJECT='q35ocr-20260921-p560'
NAMES={stage:PROJECT+'-markdown-'+stage+'-v3' for stage in ['workers','finalize','training']}

def validate_manifest(doc,stage):
    expected={('ConfigMap',NAMES[stage]+'-code'),('Job',NAMES[stage])}
    got=set()
    for item in doc['items']:
        m=item['metadata'];got.add((item['kind'],m['name']))
        if m['namespace']!=NS or m['labels'].get('project')!=PROJECT:raise ValueError('Unrelated resource in pipeline')
        if item['kind']=='Job':
            spec=item['spec']['template']['spec']
            if any(k in spec for k in ['nodeName','nodeSelector','affinity','priorityClassName']):raise ValueError('Explicit node/priority constraints are prohibited')
            if item['spec']['backoffLimit']!=0:raise ValueError('Automatic failed-job retries prohibited')
    if got!=expected:raise ValueError('Unexpected pipeline resource')

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory
    state_file=root/'pipeline-state.json';manifests={stage:json.loads((root/(stage+'-job.json')).read_text()) for stage in ['finalize','training']}
    for stage,doc in manifests.items():validate_manifest(doc,stage)
    def oc(*args,body=None):
        r=subprocess.run(['oc','-n',NS,*args],input=body,text=True,capture_output=True,timeout=45)
        if r.returncode:raise RuntimeError(r.stderr.strip())
        return r.stdout
    def optional(kind,name):
        raw=oc('get',kind,name,'--ignore-not-found','-o','json');return json.loads(raw) if raw.strip() else None
    def write(state,**fields):
        value={'state':state,'timestamp':time.time(),**fields};tmp=state_file.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(state_file)
        print('PIPELINE_STATE',json.dumps(value),flush=True)
    def wait_complete(stage,expected):
        started=time.time();last=None
        while time.time()-started<259200:
            job=optional('job',NAMES[stage])
            if not job:raise ValueError('Required pipeline job disappeared: '+stage)
            status=job.get('status',{});conditions={c['type']:c for c in status.get('conditions',[]) if c['status']=='True'}
            if 'Failed' in conditions or 'FailureTarget' in conditions:raise ValueError(stage+' failed; no restart, no GPU training submitted')
            if status.get('succeeded')==expected:return
            signature=(status.get('active',0),status.get('succeeded',0),status.get('failed',0))
            if signature!=last:write('waiting-'+stage,active=signature[0],succeeded=signature[1],failed=signature[2]);last=signature
            time.sleep(60)
        raise TimeoutError(stage+' exceeded bounded pipeline wait')
    def create(stage):
        doc=manifests[stage];existing=optional('job',NAMES[stage]);cm=next(o for o in doc['items'] if o['kind']=='ConfigMap')
        previous=optional('configmap',cm['metadata']['name'])
        if existing:
            if not previous or previous['data']!=cm['data']:raise ValueError('Existing pipeline code differs')
            if any(c['status']=='True' and c['type'] in ['Failed','FailureTarget'] for c in existing.get('status',{}).get('conditions',[])):
                raise ValueError('Existing phase failed; refusing automatic restart')
            return
        if previous:raise ValueError('ConfigMap already exists without job; inspect rather than overwrite')
        if stage=='training':
            jobs=json.loads(oc('get','jobs','-l','project='+PROJECT,'-o','json'))['items']
            if any(j.get('status',{}).get('active',0) and ('train' in j['metadata']['name'] or 'correction' in j['metadata']['name']) for j in jobs):
                raise ValueError('Another own training objective is active')
        oc('create','--dry-run=server','-f','-',body=json.dumps(doc));oc('create','-f','-',body=json.dumps(doc))
        write(stage+'-submitted',job=NAMES[stage])
    try:
        write('started',scope='one-shot conversion dependencies only; no recurring training checks')
        wait_complete('workers',4);create('finalize');wait_complete('finalize',1);create('training')
        write('training-submitted-controller-finished',job=NAMES['training'],training_started_claim=False)
    except Exception as exc:
        write('failed',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
