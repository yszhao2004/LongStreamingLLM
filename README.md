# LongStreamingLLM

## Test on MT-Bench
    cd combine
    CUDA_VISIBLE_DEVICES=0 python run.py

## Test on LongBench
### Original Model
    CUDA_VISIBLE_DEVICES=0 python pred.py
### LongLM
    CUDA_VISIBLE_DEVICES=0 python pred_long.py
### StreamingLLM
    CUDA_VISIBLE_DEVICES=0 python pred_streaming.py
### LongStreamingLLM
    CUDA_VISIBLE_DEVICES=0 python pred_longstreaming.py

#### The tests of Original Model, StreamingLLM, LongStreamingLLM are done in the environment of *transformers==4.33.0*. However, testing LongLM needs the environment of *transformers==4.38.2*. So run the following command before testing it.
    pip install transformers==4.38.2

The results are in report.pdf(Chinese version)

This project is based on the projects below:
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench/tree/main)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://github.com/datamllab/LongLM)
- [Efficient Streaming Language Models with Attention Sinks](https://github.com/mit-han-lab/streaming-llm)
