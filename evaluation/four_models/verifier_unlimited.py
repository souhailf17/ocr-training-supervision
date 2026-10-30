"""Read-only dependency/GPU check. No install, download, or model loading."""
import importlib.metadata as md,json,sys
def main():
    report={'python':sys.executable,'versions':{},'model':'baidu/Unlimited-OCR-2','model_revision':'5632329a32e103d5f1add270681e0f5c1233c6e4','installed_or_downloaded':False}
    for name in ('torch','torchvision','transformers','tokenizers','huggingface-hub','safetensors','Pillow','einops','addict','easydict','triton','numpy','requests','tqdm'):
        try:report['versions'][name]=md.version(name)
        except md.PackageNotFoundError:report['versions'][name]=None
    if report['versions']['torch']:
        try:
            import torch
            report['cuda_visible']=torch.cuda.is_available()
            if report['cuda_visible']:report['gpu']=torch.cuda.get_device_name();report['bf16']=torch.cuda.is_bf16_supported()
        except Exception as error:report['torch_import_error']=type(error).__name__
    if report['versions']['transformers']:
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention,apply_rotary_pos_emb,repeat_kv
            from transformers.cache_utils import Cache,DynamicCache
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            report['legacy_imports_required_by_official_code']=True
        except Exception as error:report['legacy_imports_required_by_official_code']=False;report['transformers_api_error']=type(error).__name__
    report['official_tested_stack']={'transformers':'4.52.1','tokenizers':'0.21.4','huggingface-hub':'0.36.0','torch':'2.13.0'}
    report['policy']='Check existing versions/imports first. If needed, use a separate environment or dependency directory; never downgrade the Qwen runtime.'
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
