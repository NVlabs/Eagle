from .fused_ops.fused_rms_norm import LigerRMSNorm
from .fused_ops.fused_rotary_pos_emb import liger_rotary_pos_emb
from .fused_ops.fused_swiglu import LigerSwiGLUMLP


def replace_liger_fused_ops():
    from transformers.models.qwen2 import modeling_qwen2
    modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    from transformers.models.llama import modeling_llama
    modeling_llama.LlamaMLP = LigerSwiGLUMLP
    modeling_llama.LlamaRMSNorm = LigerRMSNorm

    from transformers.models.qwen3 import modeling_qwen3
    modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP
    modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm
    modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    



