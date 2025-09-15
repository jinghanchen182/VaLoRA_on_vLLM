# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from statistics import mean, median
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    # parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.set_defaults(download_dir="/data/miliang/huggingface/hub")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add timing params
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs excluded from timing")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of timed runs for latency measurement")

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    warmup_runs = args.pop("warmup_runs")
    num_runs = args.pop("num_runs")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Prepare prompts
    prompts = [
        "你是一个5G基站的链路自适应专家。根据以下状态向量，决策最佳的MCS索引。状态：test_vector = \"CQI=12, RSRP=-95, SINR=20, Throughput=85, BLER=0.01\"。请只输出一个整数作为MCS索引。MCS索引：",
    ]

    # Warmup runs (excluded from timing to avoid load/compilation overhead)
    for _ in range(max(0, warmup_runs)):
        _ = llm.generate(prompts, sampling_params)

    # Timed runs
    durations = []
    last_outputs = None
    for _ in range(max(1, num_runs)):
        t0 = time.perf_counter()
        last_outputs = llm.generate(prompts, sampling_params)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    # Print last outputs and latency stats
    print("-" * 50)
    for output in last_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    print("Latency (seconds) over", len(durations), "runs")
    print("min=", f"{min(durations):.4f}",
          "median=", f"{median(durations):.4f}",
          "mean=", f"{mean(durations):.4f}",
          "max=", f"{max(durations):.4f}")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
