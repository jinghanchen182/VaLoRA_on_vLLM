# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from statistics import mean, median
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--download-dir", type=str,
                        default="/data/miliang/huggingface/hub")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                        help="e.g. '0' or '0,1'")

    return parser.parse_args()


def str_to_dtype(name: str):
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")



def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.download_dir, trust_remote_code=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.download_dir,
        torch_dtype=dtype if dtype is not None else None,
        trust_remote_code=False,
        device_map=None,
    )

    model.to(device)
    model.eval()

    prompt = "你是一个5G基站的链路自适应专家。根据以下状态向量，决策最佳的MCS索引。状态：test_vector = \"CQI=12, RSRP=-95, SINR=20, Throughput=85, BLER=0.01\"。请只输出一个整数作为MCS索引。MCS索引："

    def generate_once():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the completion after prompt for parity with vLLM print
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text

    # Warmup
    for _ in range(max(0, args.warmup_runs)):
        _ = generate_once()

    # Timed runs
    durations = []
    last_text = None
    for _ in range(max(1, args.num_runs)):
        t0 = time.perf_counter()
        last_text = generate_once()
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    # Print output and stats
    print("-" * 50)
    print(f"Prompt: {prompt!r}\nGenerated text: {last_text!r}")
    print("-" * 50)

    print("Latency (seconds) over", len(durations), "runs")
    print("min=", f"{min(durations):.4f}",
          "median=", f"{median(durations):.4f}",
          "mean=", f"{mean(durations):.4f}",
          "max=", f"{max(durations):.4f}")


if __name__ == "__main__":
    main()


