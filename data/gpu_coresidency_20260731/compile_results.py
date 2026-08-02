#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime, timezone

GIB = 1073741824
TOTAL_B = 68702699520


def stats(vals):
    v = sorted(vals)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return {"n": n, "min": round(v[0], 3), "median": round(med, 3), "max": round(v[-1], 3)}


def load(path):
    reps = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "summary" in d:
                continue
            reps.append(d)
    return reps


s1 = load("/mnt/raid0/llm/tmp/gpu_coresidency/state1.jsonl")
s2 = load("/mnt/raid0/llm/tmp/gpu_coresidency/state2.jsonl")
s3 = load("/mnt/raid0/llm/tmp/gpu_coresidency/state3.jsonl")
rc = load("/mnt/raid0/llm/tmp/gpu_coresidency/recovery.jsonl")

st1 = stats([r["decode_tps"] for r in s1])
st2 = stats([r["decode_tps"] for r in s2])
st3 = stats([r["decode_tps"] for r in s3])
strc = stats([r["decode_tps"] for r in rc])

base = st1["median"]

VRAM = {
    "gpu_total_b": TOTAL_B,
    "state1_27b_alone_b": 39405076480,
    "state2_all4_resident_idle_b": 66211344384,
    "all4_resident_idle_after_each_ran_once_b": 66999767040,
    "state3_peak_during_contention_b": 67199176704,
    "recovery_all4_resident_idle_b": 67201277952,
}
vram_gib = {k: round(v / GIB, 2) for k, v in VRAM.items()}

out = {
    "title": "GPU co-residency curiosity measurement (no gate)",
    "date_utc": datetime.now(timezone.utc).isoformat(),
    "host_gpu": "AMD Instinct MI210 gfx90a, 65520 MiB (63.98 GiB) VRAM",
    "region_lock": {"regions": ["q3"], "role": "bench", "tag": "gpu-coresidency"},
    "question": "Can the four GPU models be co-resident, and what does the 27B lose?",
    "answers": {
        "fits_at_full_262k_context": True,
        "steady_state_resident_vram_gib": vram_gib["recovery_all4_resident_idle_b"],
        "headroom_gib": round((TOTAL_B - VRAM["recovery_all4_resident_idle_b"]) / GIB, 2),
        "budget_predicted_gib": 58.30,
        "budget_error_gib": round(
            VRAM["recovery_all4_resident_idle_b"] / GIB - 58.30, 2),
        "idle_resident_cost_negligible": True,
        "idle_resident_delta_pct": round(100 * (st2["median"] - base) / base, 3),
        "active_contention_delta_pct": round(100 * (st3["median"] - base) / base, 1),
    },
    "models": {
        "qwen36_27b_dense_q8_0": {
            "path": "/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf",
            "n_ctx": 262144, "kv_type": "q8_0", "port": 8801,
            "binary_tree": "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip",
        },
        "qwen3_vl_30b_a3b_q4_k_m": {
            "path": "/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
            "mmproj": "mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf",
            "n_ctx": 16384, "port": 8802,
        },
        "whisper_large_v3_turbo": {
            "weights": "/mnt/raid0/llm/models/whisper-ggml/ggml-large-v3-turbo.bin",
            "vram_weights_mb": 1623.92, "port": 9001,
            "binary_tree": "/mnt/raid0/llm/whisper.cpp/build",
        },
        "qwen3_tts_12hz_0_6b_q8_0": {
            "talker": "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf",
            "codec": "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf",
            "resident_via": "long-lived process, --stream-by-line -o - reading a FIFO",
            "kv_cache_mb": 896, "graphs": "enabled (GGML_CUDA_DISABLE_GRAPHS unset)",
            "binary_tree": "/mnt/raid0/llm/qwentts.cpp/build",
        },
    },
    "method": {
        "decode_metric": "llama.cpp /completion timings.predicted_per_second",
        "n_predict": 256, "ignore_eos": True, "cache_prompt": False,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "seed": 42},
        "reps_per_state": 3,
        "distinct_prompt_per_rep": True,
        "warmup": "p12, discarded, not counted",
        "host_threads": "taskset -c 184-191",
        "ggml_linkage_verified": True,
    },
    "states": {
        "1_alone": {
            "description": "27B only model resident on the GPU",
            "decode_tps": st1,
            "reps": [{"prompt": r["prompt_key"], "tps": round(r["decode_tps"], 3)} for r in s1],
            "vram_gib": vram_gib["state1_27b_alone_b"],
            "delta_vs_alone_pct": 0.0,
            "gpu_busy_pct_during_decode": "99-100",
        },
        "2_others_resident_idle": {
            "description": "vision + whisper + TTS loaded and idle, 27B decoding",
            "decode_tps": st2,
            "reps": [{"prompt": r["prompt_key"], "tps": round(r["decode_tps"], 3)} for r in s2],
            "vram_gib": vram_gib["state2_all4_resident_idle_b"],
            "vram_gib_after_each_ran_once": vram_gib["all4_resident_idle_after_each_ran_once_b"],
            "delta_vs_alone_pct": round(100 * (st2["median"] - base) / base, 3),
            "gpu_busy_pct_while_idle": 0,
        },
        "3_others_active": {
            "description": "vision queries + transcriptions + syntheses running concurrently with 27B decode",
            "decode_tps": st3,
            "reps": [{"prompt": r["prompt_key"], "tps": round(r["decode_tps"], 3)} for r in s3],
            "vram_gib": vram_gib["state3_peak_during_contention_b"],
            "delta_vs_alone_pct": round(100 * (st3["median"] - base) / base, 1),
            "concurrent_work_in_window": {
                "window_s": 39.65,
                "vision_queries": 6, "vision_rate_per_s": 0.15,
                "transcriptions": 128, "transcription_rate_per_s": 3.23,
                "tts_syntheses": 19, "tts_rate_per_s": 0.48,
            },
            "note": "deliberately saturating load, not a realistic duty cycle; the GPU was pinned at 100% by the other three",
        },
        "4_recovery_control": {
            "description": "loads stopped, all four still resident and idle - control that the state-3 drop is contention, not drift",
            "decode_tps": strc,
            "reps": [{"prompt": r["prompt_key"], "tps": round(r["decode_tps"], 3)} for r in rc],
            "vram_gib": vram_gib["recovery_all4_resident_idle_b"],
            "delta_vs_alone_pct": round(100 * (strc["median"] - base) / base, 2),
        },
    },
    "side_observations": {
        "vision_latency_alone_s": 1.664,
        "vision_latency_under_contention_s": "3.9-4.0",
        "whisper_jfk_11s_clip_latency_s": 0.21,
        "tts_rtf": 0.191,
        "vram_grows_on_first_execution": "state-2 VRAM (61.66 GiB) was measured before vision/whisper/TTS had ever executed; compute buffers allocate on first run, settling at 62.59 GiB",
    },
    "caveats": [
        "Qwen3-TTS has no server mode; residency was achieved with a long-lived --stream-by-line process blocked on a FIFO. A production TTS service would need the same pattern or a server wrapper.",
        "State-3 load is a saturation stress, not a duty cycle. The -36% is a worst-case floor, not an expected steady-state loss.",
        "No gate, no threshold, no pass/fail is attached to any number here.",
    ],
}

path = "/mnt/raid0/llm/tmp/gpu_coresidency_results.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["answers"], indent=2))
print("\nwrote", path)
