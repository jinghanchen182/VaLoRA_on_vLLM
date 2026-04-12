export CUDA_VISIBLE_DEVICES=4
vllm serve LargeWorldModel/LWM-Text-Chat-1M \
--port 8071 \
--enable-lora \
--lora-modules lora1=/data/miliang/huggingface/hub/models--ohmreborn--llama-lora-7b/snapshots/d51f55b37ed14bad541db7f271017a0e9a592c77 \
lora2=/data/miliang/huggingface/hub/models--ohmreborn--llama-lora-7b/snapshots/d51f55b37ed14bad541db7f271017a0e9a592c77 \
--max-loras 2 \
--max_model_len 40000 \
--enforce_eager \
--gpu_memory_utilization 0.8 \
--no-enable-prefix-caching

# vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
# --port 8071 \
# --enable-lora \
# --lora-modules lora1=/data/miliang/huggingface/hub/models--zhen12138--qwen2.5vl7b_lora_model/snapshots/3d200f2d36098851a3b1c1a87807fe0669758453 \
# lora2=/data/miliang/huggingface/hub/models--zhen12138--qwen2.5vl7b_lora_model/snapshots/3d200f2d36098851a3b1c1a87807fe0669758453 \
# --max-loras 2 \
# --max_model_len 40000 \
# --enforce_eager \
# --gpu_memory_utilization 0.8 \
# --no-enable-prefix-caching