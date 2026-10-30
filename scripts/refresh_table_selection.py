"""Refresh an existing held-out selection against v2, keeping page IDs/splits fixed."""
import hashlib
import json
from pathlib import Path
from annotation_targets import TARGET_FORMAT, assert_no_table_placeholders


def refresh_selection(s3,bucket,summary,source: Path,destination: Path):
    if summary.get('target_format')!=TARGET_FORMAT:
        raise ValueError('Selection refresh requires the complete v2 target format')
    if source.resolve()==destination.resolve():
        raise ValueError('Refusing to overwrite the original held-out selection')
    old=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    wanted={r['id']:r for r in old}
    if len(wanted)!=len(old):
        raise ValueError('Duplicate IDs in original selection')
    found={}
    for shard in summary['shards']:
        entry=shard['manifest']
        response=s3.get_object(Bucket=bucket,Key=entry['key'])
        try: raw=response['Body'].read()
        finally: response['Body'].close()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Manifest SHA mismatch')
        for line in raw.splitlines():
            row=json.loads(line)
            if row['id'] not in wanted: continue
            prior=wanted[row['id']]
            if row['id'] in found or row['split']!=prior['split'] or row['group_id']!=prior['group_id'] or row['image']['sha256']!=prior['image']['sha256']:
                raise ValueError('Selection identity/split/image changed: '+row['id'])
            if row.get('target_format')!=TARGET_FORMAT or hashlib.sha256(row['text'].encode()).hexdigest()!=row['text_sha256']:
                raise ValueError('Invalid v2 target: '+row['id'])
            assert_no_table_placeholders(row['text'])
            found[row['id']]=row
        if len(found)==len(wanted):break
    if set(found)!=set(wanted):
        raise ValueError('V2 is missing selected pages: '+','.join(sorted(set(wanted)-set(found))))
    body=''.join(json.dumps(found[r['id']],ensure_ascii=False)+'\n' for r in old)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_text(encoding='utf-8')!=body:
        raise ValueError('Destination already exists with different data')
    temporary=destination.with_suffix(destination.suffix+'.partial')
    temporary.write_text(body,encoding='utf-8')
    temporary.replace(destination)
    return {'pages':len(old),'target_format':TARGET_FORMAT,'same_images_and_splits':True}
