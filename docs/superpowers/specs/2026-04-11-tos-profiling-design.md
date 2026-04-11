# TOS LoRA Scheduling Backend - Profiling & Optimization Design

## Context

The vLLM v1 scheduler has been modified to support three LoRA computation modes (merge/unmerge/mix) and four scheduling backends (baseline/simple/TOS/LOS). The goal is to profile TOS performance comprehensively, then optimize based on data.

### Three Computation Modes

| Mode | Model Weights | Online LoRA Computation | When Used |
|---|---|---|---|
| **merge** | `W + dW_primary` baked in | None (skip BGMV) | Single dominant LoRA |
| **unmerge** | Base `W` only | Standard BGMV shrink+expand | Multiple LoRAs, no dominant |
| **mix** | `W + dW_primary` baked in | Dual-branch: `+dW_target` and `-dW_primary` (deLoRA) | Multiple LoRAs with one dominant |

### Four Scheduling Backends

- **baseline**: Pure Punica SGMV, no merge/unmerge switching
- **simple**: Count-based majority detection, merge/unmerge only (no mix)
- **TOS**: Token-demand weighted + starvation detection (theta=30s), supports all three modes
- **LOS**: Cache-hit-ratio aware + post-batch mode decision, supports all three modes

### Test Environment

- Base model: LargeWorldModel/LWM-Text-Chat-1M (7B class)
- LoRA: rank=8 or 16, ohmreborn/llama-lora-7b
- GPU: Single GPU (CUDA_VISIBLE_DEVICES=4)
- Docker: vllm_v1
- Python env: /data/miliang/vllm/.venv

## Design

### Part 1: Infrastructure Changes

#### 1a. Environment Variable Backend Selection

Replace the hardcoded `lora_backend` string in `scheduler.py` L230-233 with an environment variable:

```python
import os
lora_backend = os.environ.get("VLLM_LORA_BACKEND", "baseline")
```

This allows switching backends without code changes: `VLLM_LORA_BACKEND=TOS vllm serve ...`

#### 1b. Merge/Unmerge Operation Timing

Add precise timing instrumentation to `gpu_model_runner.py` `_update_states()`:

- Insert `torch.cuda.synchronize()` before and after each merge/unmerge operation
- Log timing via `logger.info("[PROFILE] merge/unmerge_lora(%s) took %.4f ms")`
- This measures the actual GPU wall-clock cost of weight surgery

#### 1c. Forward Pass Mode Timing

Add per-step forward pass timing that logs:
- Total forward pass time
- Current `lora_infer_mode`
- Number of requests in batch and their LoRA distribution

### Part 2: End-to-End Benchmark Matrix

#### Server Configuration

Based on existing `vllm_server.sh`:
```bash
VLLM_LORA_BACKEND=<backend> CUDA_VISIBLE_DEVICES=4 vllm serve \
  LargeWorldModel/LWM-Text-Chat-1M \
  --port 8071 --enable-lora \
  --lora-modules lora1=<path> lora2=<path> \
  --max-loras 2 --max_model_len 40000 \
  --enforce_eager --gpu_memory_utilization 0.8 \
  --no-enable-prefix-caching
```

#### Benchmark Matrix

4 backends x 4 load scenarios = 16 runs:

| Backend | Env Var |
|---|---|
| baseline | `VLLM_LORA_BACKEND=baseline` |
| simple | `VLLM_LORA_BACKEND=simple` |
| TOS | `VLLM_LORA_BACKEND=TOS` |
| LOS | `VLLM_LORA_BACKEND=LOS` |

| Scenario | LoRA Count | Skew | Description |
|---|---|---|---|
| A | 2 | 0.5 | Uniform distribution |
| B | 2 | 0.8 | Moderate skew |
| C | 2 | 0.95 | High skew (TOS merge territory) |
| D | 2 | 1.0 | Single LoRA (merge-only control) |

#### Metrics Collected

- TTFT (Time to First Token) - P50, P90, P99
- TBT (Time Between Tokens) - mean, P90
- E2E Latency - P50, P90, P99
- Throughput (tokens/sec)

#### Benchmark Client Command

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --request-rate 6 \
  --model LargeWorldModel/LWM-Text-Chat-1M \
  --dataset-name random \
  --num-prompts 50 \
  --random-output-len 128 \
  --random-input-len 2000 \
  --lora-skew <skew> \
  --lora1-name lora1 --lora2-name lora2
```

### Part 3: Merge/Unmerge Micro-Benchmark

A standalone script that:

1. Loads the model with LoRA support
2. Performs N iterations of merge → forward → unmerge → forward
3. Measures and reports:
   - Single merge operation latency (ms)
   - Single unmerge operation latency (ms)
   - merge+unmerge round-trip latency (ms)
   - Forward pass latency in each mode: merge, unmerge, mix
   - Relative speedup of merge vs unmerge forward pass

This establishes the cost-benefit breakeven point: how many steps must you stay in merge mode before the merge operation cost is amortized by per-step forward pass savings.

### Part 4: TOS Scheduling Behavior Analysis

Parse `[TEST][TOS]` log lines to extract:

1. **Mode distribution**: fraction of steps in merge/mix/unmerge
2. **Switch frequency**: average steps between mode switches
3. **Switch pattern**: detect oscillation (merge->unmerge->merge within <5 steps)
4. **Starvation stats**: how often starvation is detected, how many requests starve
5. **Token demand ratio**: distribution of `t_dom/maxT` and `t_starve/maxT`

Output: summary statistics + time-series plots of mode decisions.

### Part 5: Automation Script

A master benchmark script (`run_tos_profile.sh`) that:

1. Iterates over backends and scenarios
2. For each combination:
   - Starts vLLM server with the right env var
   - Waits for server readiness
   - Runs the benchmark client
   - Saves results to a structured directory
   - Kills the server
3. After all runs: parses logs and generates a comparison report

Directory structure:
```
benchmark_results/
  baseline/
    skew_0.5/  metrics.json, server.log
    skew_0.8/  ...
  TOS/
    skew_0.5/  ...
  ...
  report.md   # comparison summary
```

## Implementation Order

1. Infrastructure changes (env var, timing instrumentation) - modify 2 files
2. Merge/unmerge micro-benchmark script - new file
3. Log analysis script - new file
4. Master benchmark automation script - new file
5. Run the full benchmark matrix
6. Analyze results and decide optimization direction

## Success Criteria

- Clear data showing TOS performance relative to baseline across all scenarios
- Quantified merge/unmerge switch cost in milliseconds
- Identified whether TOS mode switching provides net latency benefit
- Identified specific TOS parameters (theta, thresholds) that need tuning
- Actionable optimization recommendations backed by profiling data
