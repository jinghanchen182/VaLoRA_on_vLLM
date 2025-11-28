export LMCACHE_CONFIG_FILE=example.yaml
export CUDA_VISIBLE_DEVICES=6
export PYTHONHASHSEED=0
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8071 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'