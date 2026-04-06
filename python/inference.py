"""
    直接基于generate测试prefill和decode的时间。具体方法是：
    1. 直接测试纯prefill时间
    2. 测试prefill+decoding时间
    3. 基于上面的算出decoding时间
    4. 得到最终的TTFT和TPOT
    基于benchmark2改的，固定输入为128，输出为100，然后支持测试不同的batchsize
"""


import argparse
import csv
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@contextmanager
def inference_mode():
    """A tiny helper so we can switch between torch.inference_mode / no-op gracefully."""
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    else:  # pragma: no cover - compatibility fallback
        with torch.no_grad():
            yield


def sync_npu():
    """Synchronize NPU if available (keeps CPU timers honest)."""
    npu = getattr(torch, "npu", None)
    if npu is not None:
        npu.synchronize()


def measure_latency(fn):
    """Measure elapsed seconds for a callable using NPU events when possible."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        start = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - start

    npu.reset_peak_memory_stats()
    sync_npu()
    start_event = npu.Event(enable_timing=True)
    end_event = npu.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    sync_npu()
    elapsed = start_event.elapsed_time(end_event) / 1000.0
    return result, elapsed


def detect_device_setup(env_value=None):
    """Infer whether the job is using single or dual cards via env var."""
    value = env_value if env_value is not None else os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    value = (value or "").strip()
    if not value:
        return "unspecified", ""
    # print(value)
    devices = [token.strip() for token in value.split(",") if token.strip()]
    # print(f"devices: {devices}")
    count = len(devices) or 1
    if count == 1:
        mode = "single_card"
    elif count == 2:
        mode = "dual_card"
    else:
        mode = f"{count}_cards"
    return mode, ",".join(devices) if devices else value


RESULT_FIELDS = [
    "timestamp",
    "model",
    "bs",
    "seq_len",
    "max_new_tokens",
    "pad_mode",
    "prefill_time_s",
    "decode_time_s",
    "total_time_s",
    "output_tokens",
    "ttft_s",
    "tpot_ms",
    "throughput_tokens_per_s",
    "device_mode",
    "visible_devices",
]


def append_result_to_csv(payload, csv_path=None):
    csv_path = csv_path or Path(__file__).with_name("inference_results_bs.csv")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: payload.get(key, "") for key in RESULT_FIELDS})
    return csv_path


def safe_number(value, digits=6):
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return round(value, digits)
    return value

def build_prompt(tokenizer):
    base_prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": base_prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception as e:
            print(f"[Warning] apply_chat_template 失败，使用普通 prompt。错误信息: {e}")
    else:
        print("[Info] 当前模型不支持 chat template，使用普通 prompt。")

    return base_prompt


def pad_to_length(input_ids, tokenizer, target_len, mode):
    L_base = len(input_ids)
    if target_len < L_base:
        raise ValueError(f"seq_len ({target_len}) 不能小于原始 prompt 长度 {L_base}")

    if target_len == L_base:
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, L_base

    if mode == "repeat":
        repeat_times = (target_len + L_base - 1) // L_base
        padded = input_ids.repeat(repeat_times)[:target_len]
    elif mode == "zero":
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padding = torch.full((target_len - L_base,), pad_id, dtype=torch.long)
        padded = torch.cat([input_ids, padding])
    else:  # random
        vocab_size = tokenizer.vocab_size
        random_ids = torch.randint(0, vocab_size, (target_len - L_base,))
        padded = torch.cat([input_ids, random_ids])

    attention_mask = torch.ones_like(padded, dtype=torch.long)
    if mode == "zero":
        attention_mask[L_base:] = 0
    return padded, attention_mask, L_base


def main():
    parser = argparse.ArgumentParser(description="model 任意 seq_len 性能测试")
    parser.add_argument("--model", type=str, default="./models/BF16/Qwen3-32B",
                        help="模型路径")
    parser.add_argument("--seq_len", type=int,default=128,
                        help="目标输入序列长度（包含系统/用户 prompt + padding）")
    # parser.add_argument("--max_new_tokens", type=int, default=100,
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="生成 token 数")
    parser.add_argument("--pad_mode", choices=["repeat", "zero", "random"], default="repeat",
                        help="填充方式: repeat(重复原始 prompt), zero(全 0), random(随机 token)")
    parser.add_argument("--batch_size", type=int, default=1,
                    help="批大小")
    args = parser.parse_args()

    # --------------------- 加载模型 & tokenizer ---------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        # trust_remote_code=True
    )
    model.eval()

    # --------------------- 构造基础 prompt ---------------------
    text = build_prompt(tokenizer)

    # --------------------- 编码成 token ids ---------------------
    input_ids = tokenizer([text], return_tensors="pt").input_ids[0]
    input_ids, attention_mask, L_base = pad_to_length(input_ids, tokenizer, args.seq_len, args.pad_mode)
    print(f"原始 prompt 长度: {L_base} tokens")

    # 转为 batch（batch_size=1）
    # input_ids = input_ids.unsqueeze(0).to(model.device)
    # attention_mask = attention_mask.unsqueeze(0).to(model.device)
    bs = args.batch_size
    input_ids = input_ids.unsqueeze(0).repeat(bs, 1).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).repeat(bs, 1).to(model.device)


    print(f"最终输入长度: {input_ids.shape[1]} tokens")

    # --------------------- Prefill 计时 ---------------------
    def _prefill():
        with inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask)
    # time.sleep(10)
    # warmup
    for _ in range(3):
        _prefill()

    _, prefill_time = measure_latency(_prefill)
    print(f"Prefill 耗时: {prefill_time:.3f}s")

    # --------------------- Prefill + Decode 计时 ---------------------
    def _generate():
        with inference_mode():
            return model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
            )

    generated_ids, total_time = measure_latency(_generate)
    total_token_count = generated_ids.shape[1]
    output_token_count = max(total_token_count - input_ids.shape[1], 0)
    decode_time = max(total_time - prefill_time, 0.0)
    per_token_latency = decode_time / output_token_count if output_token_count else float("nan")
    per_token_latency_ms = per_token_latency * 1000 if output_token_count else float("nan")
    tokens_per_second = output_token_count / decode_time if output_token_count and decode_time > 0 else float("nan")
    ttft = prefill_time + (per_token_latency if output_token_count else 0.0)

    print("\n========= 性能指标 =========")
    print(f"Prefill + Decode 总耗时: {total_time:.3f}s")
    print(f"Decode 耗时(推算): {decode_time:.3f}s")
    print(f"输出 token 数: {output_token_count}")
    if output_token_count:
        print(f"TTFT (约): {ttft:.3f}s  # Prefill + 首个 token 推断")
        if math.isfinite(per_token_latency_ms):
            print(f"TPOT: {per_token_latency_ms:.2f}ms/token")
        else:
            print("TPOT: 无法计算（decode_time <= 0 或输出 token 为 0）")
        if math.isfinite(tokens_per_second):
            print(f"吞吐: {tokens_per_second:.2f} tokens/s")
        else:
            print("吞吐: 无法计算（decode_time <= 0）")
    else:
        print("未生成新 token，TTFT/TPOT 无法计算。")

    device_mode, visible_devices = detect_device_setup()
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "bs": args.batch_size,
        "seq_len": args.seq_len,
        "max_new_tokens": args.max_new_tokens,
        "pad_mode": args.pad_mode,
        "prefill_time_s": safe_number(prefill_time),
        "decode_time_s": safe_number(decode_time),
        "total_time_s": safe_number(total_time),
        "output_tokens": output_token_count,
        "ttft_s": safe_number(ttft),
        "tpot_ms": safe_number(per_token_latency_ms),
        "throughput_tokens_per_s": safe_number(tokens_per_second),
        "device_mode": device_mode,
        "visible_devices": visible_devices,
    }
    csv_path = append_result_to_csv(payload)
    print(f"[Info] 已将测评结果追加至 {csv_path.resolve()}")




if __name__ == "__main__":
    main()


