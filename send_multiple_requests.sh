python -m vllm.entrypoints.cli.main bench serve \
    --request-rate 6 \
    --model LargeWorldModel/LWM-Text-Chat-1M  \
    --port 8071 \
    --dataset-name random \
    --goodput ttft:200 \
    --num-prompts 50 \
    --random_output_len 1 \
    --random-input-len 2000 \
    --lora-skew 0.8 \
    --lora1-name lora1 \
    --lora2-name lora2

# python -m vllm.entrypoints.cli.main bench serve \
#     --request-rate 1 \
#     --model Qwen/Qwen2.5-VL-7B-Instruct  \
#     --port 8071 \
#     --dataset-name random \
#     --goodput ttft:200 \
#     --num-prompts 10 \
#     --random_output_len 10 \
#     --random-input-len 2000 \
#     --lora-skew 1 \
#     --lora1-name lora1 \
#     --lora2-name lora2