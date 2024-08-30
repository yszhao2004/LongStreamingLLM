import longstreaming.long_llama_self_extend_patch as LlamaSE
from longstreaming.long_modify_utils import modify_method_of_instance
from functools import partial
from longstreaming.stm_kv_cache import StartRecentKVCache

def enable_streaming_llm(model, start_size, recent_size):
    k_seq_dim = v_seq_dim = 2
    self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=8, group_size_2=1024)
    modify_method_of_instance(model, "LlamaAttention", "forward", self_extend_forward)

    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
