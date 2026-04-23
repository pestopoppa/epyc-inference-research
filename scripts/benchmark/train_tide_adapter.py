#!/usr/bin/env python3
"""Train bottleneck MLP adapter for TIDE early exit.

Maps layer-32 hidden states → layer-64 hidden states.
Architecture: Linear(5120, bottleneck) → ReLU → Linear(bottleneck, 5120) + residual

The adapter replaces missing layers 33-64. After the adapter,
the normal output_norm + LM head runs as usual.
"""

import json
import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

DATA_DIR = Path("/tmp/tide_27b_retrain")
OUT_DIR = DATA_DIR  # save adapters next to training data

# Load metadata
with open(DATA_DIR / "meta.json") as f:
    meta = json.load(f)

n_samples = meta["n_samples"]
seq_len = meta["seq_len"]
n_embd = meta["n_embd"]
total_vectors = n_samples * seq_len

print(f"Loading data: {n_samples} samples × {seq_len} tokens = {total_vectors} vectors, dim={n_embd}")

# Load hidden states
input_data = np.fromfile(DATA_DIR / "hidden_layer_032.bin", dtype=np.float32)
target_data = np.fromfile(DATA_DIR / "hidden_layer_064.bin", dtype=np.float32)

input_data = input_data.reshape(total_vectors, n_embd)
target_data = target_data.reshape(total_vectors, n_embd)

print(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")

# Quick stats
cos_raw = np.sum(input_data[:100] * target_data[:100], axis=1) / (
    np.linalg.norm(input_data[:100], axis=1) * np.linalg.norm(target_data[:100], axis=1) + 1e-8
)
print(f"Raw cosine similarity (layer32 vs layer64): mean={cos_raw.mean():.4f}, std={cos_raw.std():.4f}")

# Train/val split (80/20 by sample, not by token)
n_train_samples = int(n_samples * 0.8)
n_train = n_train_samples * seq_len
n_val = total_vectors - n_train

X_train = torch.tensor(input_data[:n_train], dtype=torch.float32)
Y_train = torch.tensor(target_data[:n_train], dtype=torch.float32)
X_val = torch.tensor(input_data[n_train:], dtype=torch.float32)
Y_val = torch.tensor(target_data[n_train:], dtype=torch.float32)

print(f"Train: {n_train} vectors, Val: {n_val} vectors")

# Normalize inputs for stable training
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0).clamp(min=1e-6)
Y_mean = Y_train.mean(dim=0)
Y_std = Y_train.std(dim=0).clamp(min=1e-6)

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
Y_train_norm = (Y_train - Y_mean) / Y_std
Y_val_norm = (Y_val - Y_mean) / Y_std


class BottleneckAdapter(nn.Module):
    def __init__(self, n_embd, bottleneck_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, n_embd),
        )
        # Residual connection: adapter learns the DELTA
        # layer64 ≈ layer32 + adapter(layer32)
        self.use_residual = True

    def forward(self, x):
        if self.use_residual:
            return x + self.net(x)
        return self.net(x)


# Sweep bottleneck dims
bottleneck_dims = [128, 256, 512]
results = {}

for bottleneck_dim in bottleneck_dims:
    print(f"\n{'='*60}")
    print(f"Training bottleneck_dim={bottleneck_dim}")
    n_params = n_embd * bottleneck_dim + bottleneck_dim + bottleneck_dim * n_embd + n_embd
    print(f"Parameters: {n_params:,}")

    model = BottleneckAdapter(n_embd, bottleneck_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    batch_size = 2048
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(100):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            x = X_train_norm[idx]
            # Target is delta in normalized space (residual learning)
            y = Y_train_norm[idx] - X_train_norm[idx]

            pred = model.net(x)  # raw network output
            loss = nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_norm)  # includes residual
            val_pred_denorm = val_pred * Y_std + Y_mean
            val_target = Y_val

            # MSE in original space
            val_mse = nn.functional.mse_loss(val_pred_denorm, val_target).item()

            # Cosine similarity in original space
            cos_sim = nn.functional.cosine_similarity(val_pred_denorm, val_target, dim=1)
            val_cos_mean = cos_sim.mean().item()
            val_cos_min = cos_sim.min().item()

        avg_train_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f} val_mse={val_mse:.6f} "
                  f"val_cos={val_cos_mean:.4f} (min={val_cos_min:.4f}) lr={scheduler.get_last_lr()[0]:.6f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # Final validation metrics
    with torch.no_grad():
        val_pred = model(X_val_norm)
        val_pred_denorm = val_pred * Y_std + Y_mean
        cos_sim = nn.functional.cosine_similarity(val_pred_denorm, Y_val, dim=1)

    results[bottleneck_dim] = {
        "val_cos_mean": cos_sim.mean().item(),
        "val_cos_std": cos_sim.std().item(),
        "val_cos_min": cos_sim.min().item(),
        "val_mse": best_val_loss,
        "model": model,
        "best_state": best_state,
    }

    print(f"\n  FINAL: cos_mean={cos_sim.mean():.4f} cos_std={cos_sim.std():.4f} "
          f"cos_min={cos_sim.min():.4f} mse={best_val_loss:.6f}")

# Pick best
print(f"\n{'='*60}")
print("SUMMARY:")
best_dim = max(results, key=lambda d: results[d]["val_cos_mean"])
for dim in bottleneck_dims:
    r = results[dim]
    marker = " ← BEST" if dim == best_dim else ""
    print(f"  dim={dim:4d}: cos={r['val_cos_mean']:.4f}±{r['val_cos_std']:.4f} "
          f"(min={r['val_cos_min']:.4f}) mse={r['val_mse']:.6f}{marker}")


def save_adapter(model, dim, path, x_mean, x_std, y_mean, y_std, n_embd):
    """Save adapter in binary format for C++ loading.

    Format:
      Header: magic(4) + version(4) + n_embd(4) + bottleneck(4) + use_residual(4) + has_norm(4)
      Norm stats: X_mean(n_embd) + X_std(n_embd) + Y_mean(n_embd) + Y_std(n_embd)
      Weights: W1(bottleneck×n_embd) + b1(bottleneck) + W2(n_embd×bottleneck) + b2(n_embd)
    """
    with open(path, "wb") as f:
        f.write(b"TIDE")
        f.write(struct.pack("<i", 1))  # version
        f.write(struct.pack("<i", n_embd))
        f.write(struct.pack("<i", dim))
        f.write(struct.pack("<i", 1 if model.use_residual else 0))
        f.write(struct.pack("<i", 1))  # has_norm

        # Normalization stats
        f.write(x_mean.numpy().tobytes())
        f.write(x_std.numpy().tobytes())
        f.write(y_mean.numpy().tobytes())
        f.write(y_std.numpy().tobytes())

        # Network weights
        f.write(model.net[0].weight.data.numpy().tobytes())  # W1: (bottleneck, n_embd)
        f.write(model.net[0].bias.data.numpy().tobytes())    # b1: (bottleneck,)
        f.write(model.net[2].weight.data.numpy().tobytes())  # W2: (n_embd, bottleneck)
        f.write(model.net[2].bias.data.numpy().tobytes())    # b2: (n_embd,)


# Save all adapters
for dim in bottleneck_dims:
    r = results[dim]
    m = r["model"]
    m.load_state_dict(r["best_state"])
    p = OUT_DIR / f"adapter_b{dim}.bin"
    save_adapter(m, dim, p, X_mean, X_std, Y_mean, Y_std, n_embd)
    size_mb = p.stat().st_size / 1e6
    marker = " ← BEST" if dim == best_dim else ""
    print(f"Saved: {p} ({size_mb:.1f} MB){marker}")

print(f"\nBest architecture: Linear({n_embd}, {best_dim}) → ReLU → Linear({best_dim}, {n_embd}) + residual")
print(f"Total parameters: {sum(p.numel() for p in results[best_dim]['model'].parameters()):,}")
