# LongStreamingLLM

## Test on MT-Bench
    cd combine
    CUDA_VISIBLE_DEVICES=0 python run.py

## Test on LongBench
### original model
    CUDA_VISIBLE_DEVICES=0 python pred.py
### LongLM
    CUDA_VISIBLE_DEVICES=0 python pred_long.py
### StreamingLLM
    CUDA_VISIBLE_DEVICES=0 python pred_streaming.py
### LongStreamingLLM
    CUDA_VISIBLE_DEVICES=0 python pred_longstreaming.py
    
This project is based on the projects below:
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench/tree/main)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://github.com/datamllab/LongLM)
- [Efficient Streaming Language Models with Attention Sinks](https://github.com/mit-han-lab/streaming-llm)
