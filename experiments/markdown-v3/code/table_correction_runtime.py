"""Shared exact encoder/model for the isolated tables-only corrective phase."""
import hashlib,io
from pathlib import Path
from annotation_targets import TABLE_OCR_PROMPT,TARGET_FORMAT,assert_no_table_placeholders

PARENT_SHA='e32116d77a324a8b970503e23440dbc83434a21b171a2ec3bcb87d14e6c503e6'
SOURCE_DATA_SHA='89e75461a805005be53d18e772df31422f06a8fc9f3bbd32f636350cc5bd153b'
PARENT_STEP=14355
MAX_SEQUENCE=8192

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def encode(processor,row,raw=None,pin=False):
    import torch
    from PIL import Image
    assert row['target_format']==TARGET_FORMAT
    assert_no_table_placeholders(row['text'])
    if raw is None:raw=Path(row['local_image']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['image']['sha256']
    assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
    with Image.open(io.BytesIO(raw)) as im:image=im.convert('RGB')
    image.thumbnail((1024,1024))
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':TABLE_OCR_PROMPT}]}]
    kwargs=dict(tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    prompt=processor.apply_chat_template(messages,add_generation_prompt=True,**kwargs)
    encoded=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':row['text']}]}],add_generation_prompt=False,**kwargs)
    n=prompt['input_ids'].shape[1]
    assert torch.equal(prompt['input_ids'],encoded['input_ids'][:,:n]),'Assistant boundary differs'
    if encoded['input_ids'].shape[1]>MAX_SEQUENCE:raise ValueError('sequence_exceeds_8192_no_truncation')
    labels=encoded['input_ids'].clone();labels[:,:n]=-100;encoded['labels']=labels
    assert int((labels[:,1:]!=-100).sum())>0
    if pin:encoded={k:v.pin_memory() for k,v in encoded.items()}
    return encoded

def load_parent(model_dir,checkpoint,device):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from lora import LowRankLinear
    assert digest(checkpoint)==PARENT_SHA
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['clock']['step_index']==PARENT_STEP
    assert state['data_complete_sha256']==SOURCE_DATA_SHA
    adapters=state['adapter'];assert len(adapters)==256
    assert all(torch.isfinite(t).all() for t in adapters.values())
    model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    model.requires_grad_(False);count=0
    for name,module in list(model.named_modules()):
        if isinstance(module,torch.nn.Linear) and 'language_model.layers.' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}:
            parent,attribute=name.rsplit('.',1)
            setattr(model.get_submodule(parent),attribute,LowRankLinear(module,rank=16,alpha=32,compute_dtype='fp32'));count+=1
    assert count==128
    used=set()
    for name,p in model.named_parameters():
        if p.requires_grad:
            key=next((prefix+name for prefix in ('module.hf.','hf.') if prefix+name in adapters),None)
            assert key is not None and adapters[key].shape==p.shape
            with torch.no_grad():p.copy_(adapters[key])
            assert torch.equal(p.detach().cpu(),adapters[key]);used.add(key)
    assert used==set(adapters)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    return model
