#!/usr/bin/env python3
"""Per-call latency of OpenVLA-7B (arXiv 2406.09246) with synthetic inputs, for the on-device latency
table. OpenVLA emits one action per call as a string of action tokens (one token per action dimension),
so the cost is a prefill over the image patches + prompt followed by action_dim greedy decode steps.

    usage: bench_openvla_latency.py <checkpoint dir> <n calls> <out.json>

We time two variants: the released 7-D Bridge action head (predict_action, unnorm_key=bridge_orig) and a
13-token generation matching USIM's 13-D action (the paper fine-tuned OpenVLA on USIM; the number of
decode steps, not the fine-tuning, sets the latency). Needs transformers==4.40.1 / timm==0.9.10, the
versions OpenVLA's remote code targets."""
import json
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def main() -> None:
    ckpt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    dev = "cuda:0"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, attn_implementation="sdpa", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(dev).eval()
    load_s = time.time() - t0
    params_b = sum(p.numel() for p in vla.parameters()) / 1e9
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to go to the water tower?\nOut:"
    inputs = processor(prompt, image).to(dev, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()

    def timed(fn, k):
        ts = []
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        for _ in range(k):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        return ts

    with torch.inference_mode():
        # released head: 7 Bridge action tokens
        ts7 = timed(lambda: vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False), n)
        # USIM-sized head: 13 action tokens
        ids = inputs["input_ids"]
        if not torch.all(ids[:, -1] == 29871):   # predict_action appends the llama space token; do the same
            ids = torch.cat((ids, torch.tensor([[29871]], device=ids.device, dtype=ids.dtype)), dim=1)
        gen_inputs = {**inputs, "input_ids": ids,
                      "attention_mask": torch.ones_like(ids)}
        ts13 = timed(lambda: vla.generate(**gen_inputs, max_new_tokens=13, do_sample=False,
                                          pad_token_id=processor.tokenizer.pad_token_id), n)
    res = {
        "model": ckpt, "policy_type": "openvla", "params_B": params_b, "load_s": load_s,
        "median_ms": statistics.median(ts13), "p95_ms": float(np.percentile(ts13, 95)),
        "median_ms_7tok": statistics.median(ts7), "p95_ms_7tok": float(np.percentile(ts7, 95)),
        "n": n, "action_tokens": 13, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "device": torch.cuda.get_device_name(0), "attn": "sdpa", "dtype": "bf16",
        "image": "224x224 synthetic", "prompt": prompt,
    }
    print("[bench-openvla]", json.dumps(res))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
