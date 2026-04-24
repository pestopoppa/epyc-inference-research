#!/usr/bin/env python3
"""
TIDE Calibration Script — Record hidden states for router training.

Runs N samples through the model, recording hidden states at checkpoint layers
(every CHECKPOINT_INTERVAL layers). Stores full hidden states for flexible
experimentation with router architectures and thresholds.

Output:
  - Hidden states: {output_dir}/hidden_states_layer_{L}.npy for each checkpoint layer
  - Metadata: {output_dir}/calibration_meta.json

Requirements:
  - llama-cpp-python (pip install llama-cpp-python)
  - numpy
  - A running llama-server with the target model

Usage:
  python calibrate_tide_router.py --model-path /path/to/model.gguf --output-dir /path/to/output
  python calibrate_tide_router.py --server http://127.0.0.1:8093 --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Attempt to import llama_cpp for direct model loading
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


# --- Configuration ---
DEFAULT_N_SAMPLES = 2000
DEFAULT_SEQ_LEN = 512
DEFAULT_CHECKPOINT_INTERVAL = 4  # Record every 4th layer
DEFAULT_DATASET = "wikitext"  # placeholder — uses built-in text


def load_calibration_text(n_samples: int, seq_len: int) -> list[str]:
    """Load calibration text samples.

    Uses a mix of sources to cover diverse token distributions:
    - Wikipedia-style prose (varied vocabulary)
    - Code (structured patterns)
    - Technical writing (domain-specific)
    """
    # Try to load WikiText from HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        texts = []
        current_text = ""
        for item in ds:
            text = item.get("text", "")
            if len(text.strip()) < 50:
                continue
            current_text += text + " "
            if len(current_text) >= seq_len * 4:  # ~4 chars per token
                texts.append(current_text[:seq_len * 5])
                current_text = ""
                if len(texts) >= n_samples:
                    break
        print(f"Loaded {len(texts)} samples from WikiText-103", flush=True)
        return texts
    except Exception as e:
        print(f"WikiText unavailable ({e}), using synthetic calibration text", flush=True)

    # Fallback: generate diverse calibration text
    texts = []
    base_texts = [
        "The transformer architecture has revolutionized natural language processing since its introduction in 2017. "
        "Self-attention mechanisms allow the model to weigh the importance of different parts of the input sequence. "
        "Multi-head attention provides multiple representation subspaces, enabling the model to attend to different "
        "types of relationships simultaneously. The feed-forward networks in each layer provide nonlinear transformations "
        "that increase the model's capacity to represent complex functions.",

        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n    return b\n\ndef factorial(n: int) -> int:\n    result = 1\n    for i in range(2, n + 1):\n"
        "        result *= i\n    return result\n\nclass BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n",

        "In quantum mechanics, the wave function describes the quantum state of a system. The Schrödinger equation "
        "governs how the wave function evolves over time. When a measurement is performed, the wave function collapses "
        "to an eigenstate of the measurement operator. The Born rule gives the probability of measuring a particular "
        "eigenvalue as the squared magnitude of the corresponding amplitude.",
    ]
    for i in range(n_samples):
        text = base_texts[i % len(base_texts)] * 10  # Repeat to fill seq_len
        texts.append(text[:seq_len * 5])
    return texts


def calibrate_with_server(
    server_url: str,
    n_samples: int,
    seq_len: int,
    checkpoint_interval: int,
    output_dir: Path,
):
    """Calibrate using a running llama-server with hidden state extraction.

    NOTE: This requires a server that supports hidden state extraction via a
    custom endpoint. If not available, falls back to the direct model approach.
    """
    raise NotImplementedError(
        "Server-based calibration requires custom /v1/hidden_states endpoint. "
        "Use --model-path for direct model loading instead."
    )


def calibrate_with_model(
    model_path: str,
    n_samples: int,
    seq_len: int,
    checkpoint_interval: int,
    output_dir: Path,
    n_threads: int = 48,
    n_gpu_layers: int = 0,
):
    """Calibrate by loading model directly via llama-cpp-python.

    Records hidden states at each checkpoint layer for all tokens in all samples.
    """
    if not LLAMA_CPP_AVAILABLE:
        print("ERROR: llama-cpp-python not installed. Install with:")
        print("  pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading model: {model_path}", flush=True)
    print(f"Config: {n_samples} samples, {seq_len} tokens/sample, checkpoints every {checkpoint_interval} layers", flush=True)

    # Load model with hidden state extraction enabled
    model = Llama(
        model_path=model_path,
        n_ctx=seq_len + 64,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    # Get model dimensions
    n_layers = model.n_layer()  # type: ignore
    n_embd = model.n_embd()  # type: ignore
    checkpoint_layers = list(range(checkpoint_interval, n_layers + 1, checkpoint_interval))
    n_checkpoints = len(checkpoint_layers)

    print(f"Model: {n_layers} layers, {n_embd} hidden dim", flush=True)
    print(f"Checkpoint layers: {checkpoint_layers} ({n_checkpoints} checkpoints)", flush=True)

    # Calculate storage
    total_bytes = n_samples * seq_len * n_checkpoints * n_embd * 4  # float32
    print(f"Storage needed: {total_bytes / 1e9:.1f} GB", flush=True)

    # Load calibration text
    texts = load_calibration_text(n_samples, seq_len)

    # Allocate output arrays (one file per checkpoint layer)
    os.makedirs(output_dir, exist_ok=True)

    # Memory-mapped files for incremental writes
    hidden_files = {}
    for i, layer in enumerate(checkpoint_layers):
        path = output_dir / f"hidden_states_layer_{layer:03d}.npy"
        arr = np.lib.format.open_memmap(
            str(path), mode='w+', dtype=np.float32,
            shape=(n_samples, seq_len, n_embd),
        )
        hidden_files[layer] = arr

    # Process samples
    start_time = time.time()
    for sample_idx in range(n_samples):
        if sample_idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = sample_idx / elapsed if elapsed > 0 else 0
            eta = (n_samples - sample_idx) / rate if rate > 0 else 0
            print(f"  [{sample_idx}/{n_samples}] {rate:.1f} samples/s, ETA {eta:.0f}s", flush=True)

        # Tokenize
        text = texts[sample_idx]
        tokens = model.tokenize(text.encode("utf-8"), add_bos=True)[:seq_len]
        n_tokens = len(tokens)

        # Run forward pass with hidden state extraction
        # llama-cpp-python doesn't natively expose per-layer hidden states.
        # We need to use the low-level eval with output_hidden_states=True
        # This requires a patched version or custom binding.
        #
        # FALLBACK: Use the model's logits interface and reconstruct from
        # the embedding output at each layer via a custom callback.
        #
        # For now, we use the eval() method which processes the full sequence
        # and we extract the final hidden state (pre-LM-head).
        # Full per-layer extraction requires a C-level hook.

        model.reset()
        model.eval(tokens)

        # NOTE: llama-cpp-python only exposes the final layer's output by default.
        # For full per-layer hidden states, we need either:
        # 1. A custom llama.cpp build with per-layer output hooks (preferred)
        # 2. Running the model N times, each with n_layer_exit set to a different checkpoint
        #
        # Approach 2 is used here: run forward pass multiple times with different exit points.
        # This is N_checkpoints × slower but doesn't require C code changes for Phase 1.
        for ckpt_idx, layer in enumerate(checkpoint_layers):
            # Set exit layer and run
            # This uses our existing n_layer_exit infrastructure
            model._model.n_layer_exit = layer  # type: ignore
            model.reset()
            model.eval(tokens)

            # Extract the hidden state at the exit point
            # After eval with n_layer_exit, the output embeddings reflect state at that layer
            embeddings = model.embed(tokens)  # shape: (n_tokens, n_embd)
            if embeddings is not None and len(embeddings) > 0:
                # Pad or truncate to seq_len
                actual_len = min(len(embeddings), seq_len)
                hidden_files[layer][sample_idx, :actual_len, :] = embeddings[:actual_len]

        # Reset n_layer_exit for next sample
        model._model.n_layer_exit = 0  # type: ignore

    elapsed = time.time() - start_time
    print(f"\nCalibration complete: {n_samples} samples in {elapsed:.0f}s ({n_samples/elapsed:.1f} samples/s)", flush=True)

    # Flush memory-mapped files
    for arr in hidden_files.values():
        del arr

    # Save metadata
    meta = {
        "model_path": model_path,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_embd": n_embd,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_layers": checkpoint_layers,
        "n_checkpoints": n_checkpoints,
        "calibration_time_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "storage_gb": total_bytes / 1e9,
    }
    meta_path = output_dir / "calibration_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}", flush=True)

    return meta


def train_routers(
    output_dir: Path,
    threshold: float = 0.98,
    hidden_dim: int = 128,
    epochs: int = 10,
    batch_size: int = 1024,
):
    """Train router MLPs from stored hidden states.

    For each pair of consecutive checkpoint layers, trains a binary classifier:
    - Input: hidden state at layer L
    - Target: 1 if cos_sim(h[L], h[L+interval]) > threshold, else 0

    Saves router weights as a single .tide.bin file.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    meta_path = output_dir / "calibration_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    checkpoint_layers = meta["checkpoint_layers"]
    n_embd = meta["n_embd"]
    n_samples = meta["n_samples"]
    seq_len = meta["seq_len"]

    print(f"Training routers (threshold={threshold}, hidden={hidden_dim})", flush=True)
    print(f"  Checkpoints: {checkpoint_layers}", flush=True)

    routers = {}

    for i in range(len(checkpoint_layers) - 1):
        layer_a = checkpoint_layers[i]
        layer_b = checkpoint_layers[i + 1]

        print(f"  Router {layer_a}→{layer_b}: ", end="", flush=True)

        # Load hidden states for this pair
        h_a = np.load(str(output_dir / f"hidden_states_layer_{layer_a:03d}.npy"), mmap_mode='r')
        h_b = np.load(str(output_dir / f"hidden_states_layer_{layer_b:03d}.npy"), mmap_mode='r')

        # Reshape to (n_samples * seq_len, n_embd)
        h_a_flat = h_a.reshape(-1, n_embd)
        h_b_flat = h_b.reshape(-1, n_embd)

        # Compute cosine similarity
        norm_a = np.linalg.norm(h_a_flat, axis=1, keepdims=True)
        norm_b = np.linalg.norm(h_b_flat, axis=1, keepdims=True)
        cos_sim = np.sum(h_a_flat * h_b_flat, axis=1) / (norm_a.squeeze() * norm_b.squeeze() + 1e-8)

        # Binary labels
        labels = (cos_sim > threshold).astype(np.float32)
        convergence_rate = labels.mean()
        print(f"convergence={convergence_rate:.1%}, ", end="", flush=True)

        # Train MLP: Linear(n_embd, hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -> Sigmoid
        X = torch.tensor(h_a_flat, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        router = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        router.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = router(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate
        router.eval()
        with torch.no_grad():
            preds = torch.sigmoid(router(X)) > 0.5
            accuracy = (preds.squeeze() == y.squeeze()).float().mean().item()

        print(f"accuracy={accuracy:.3f}", flush=True)
        routers[layer_a] = router.state_dict()

    # Save all routers to a single file
    tide_path = output_dir / "routers.tide.bin"
    torch.save({
        "routers": routers,
        "meta": {
            "threshold": threshold,
            "hidden_dim": hidden_dim,
            "n_embd": n_embd,
            "checkpoint_layers": checkpoint_layers,
        }
    }, str(tide_path))
    print(f"\nRouters saved to {tide_path} ({tide_path.stat().st_size / 1024:.1f} KB)", flush=True)

    return routers


def main():
    parser = argparse.ArgumentParser(description="TIDE Calibration — record hidden states and train routers")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model file (direct loading)")
    parser.add_argument("--server", type=str, help="Server URL for extraction (not yet supported)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for hidden states")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Tokens per sample")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="Record every N layers")
    parser.add_argument("--threads", type=int, default=48, help="CPU threads for model inference")
    parser.add_argument("--train-only", action="store_true", help="Skip calibration, train from existing data")
    parser.add_argument("--threshold", type=float, default=0.98, help="Cosine similarity threshold for convergence")
    parser.add_argument("--router-hidden", type=int, default=128, help="Router MLP hidden dimension")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if not args.train_only:
        if args.server:
            calibrate_with_server(args.server, args.n_samples, args.seq_len,
                                  args.checkpoint_interval, output_dir)
        elif args.model_path:
            calibrate_with_model(args.model_path, args.n_samples, args.seq_len,
                                 args.checkpoint_interval, output_dir,
                                 n_threads=args.threads)
        else:
            print("ERROR: Provide either --model-path or --server")
            sys.exit(1)

    # Train routers
    if (output_dir / "calibration_meta.json").exists():
        train_routers(output_dir, threshold=args.threshold, hidden_dim=args.router_hidden)
    else:
        print("No calibration data found. Run without --train-only first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
