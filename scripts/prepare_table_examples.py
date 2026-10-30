"""Build separate image-only inference inputs and complete scoring references."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from annotation_targets import resolve_annotation, TABLE_OCR_PROMPT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotations',nargs='+',type=Path,required=True)
    parser.add_argument('--images',nargs='*',type=Path,default=[])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    references=[];images={};proof={}
    for path in args.annotations:
        raw=path.read_bytes();annotation=json.loads(raw);resolved=resolve_annotation(annotation)
        ident=path.stem
        proof[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        target=args.output_dir/'targets'/(ident+'.md');target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_text(encoding='utf-8')!=resolved.text:
            raise ValueError('Refusing to overwrite a different corrected target')
        target.write_text(resolved.text,encoding='utf-8')
        references.append({'id':ident,'annotation':annotation,'corrected_reference':resolved.text,'audit':resolved.audit,
                           'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()})
        image=path.with_suffix('.png')
        if image.is_file():images[ident]=image
    for path in args.images:
        if path.stem in images and images[path.stem].read_bytes()!=path.read_bytes():
            raise ValueError('Conflicting images for '+path.stem)
        images[path.stem]=path
    records=[]
    for ident,path in sorted(images.items()):
        raw=path.read_bytes()
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:width,height=image.size
        reference=next((r for r in references if r['id']==ident),None)
        if reference:
            dimensions=reference['annotation'].get('dimensions',{})
            if (width,height)!=(dimensions.get('width'),dimensions.get('height')):
                raise ValueError('Image/annotation dimensions do not match '+ident)
        target=args.output_dir/'images'/(ident+'.png');target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_bytes()!=raw:raise ValueError('Image output already differs')
        target.write_bytes(raw)
        proof[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        # Deliberately no reference/table text in this model-input record.
        records.append({'id':ident,'image_path':'images/'+target.name,'image_sha256':hashlib.sha256(raw).hexdigest(),
                        'width':width,'height':height,'reference_available':reference is not None})
    summary={'target_tables':sum(r['audit']['table_count'] for r in references),
             'target_cells':sum(r['audit']['cell_count'] for r in references),
             'annotations':len(references),'inference_images':len(records),
             'scorable_pairs':sum(r['reference_available'] for r in records),
             'annotation_without_image':[r['id'] for r in references if r['id'] not in images],
             'image_without_annotation':[r['id'] for r in records if not r['reference_available']],
             'model_was_run':False,'training_was_run':False,'source_files_unchanged':proof}
    for name,obj in [('references.json',{'records':references}),
                     ('inference-inputs.json',{'prompt':TABLE_OCR_PROMPT,'records':records,'ground_truth_in_model_inputs':False,
                                               'purpose':'Diagnostic samples; training overlap and shard identity are not certified.'}),
                     ('verification.json',summary)]:
        path=args.output_dir/name;body=json.dumps(obj,ensure_ascii=False,indent=2)
        if path.exists() and path.read_text(encoding='utf-8')!=body:raise ValueError('Refusing to replace different '+name)
        path.write_text(body,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
