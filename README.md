# LongStreamingLLM

## test on MT-Bench
    cd combine
    CUDA_VISIBLE_DEVICES=0 python run.py

## test on LongBench
    CUDA_VISIBLE_DEVICES=0 python pred.py
    CUDA_VISIBLE_DEVICES=0 python pred_long.py
    CUDA_VISIBLE_DEVICES=0 python pred_streaming.py
    CUDA_VISIBLE_DEVICES=0 python pred_longstreaming.py
    
