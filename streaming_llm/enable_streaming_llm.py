from streaming_llm.kv_cache import StartRecentKVCache


def enable_streaming_llm(model, start_size, recent_size):
    k_seq_dim = v_seq_dim = 2
    from streaming_llm.pos_shift.modify_llama import (
        enable_llama_pos_shift_attention,
    )

    enable_llama_pos_shift_attention(model)

    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
