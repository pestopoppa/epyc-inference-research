#!/usr/bin/env python3
"""RT-1 results-dir validation: shard accounting, window coverage, row/phase geometry, id sanity and an independent
recomputation of the EOR/IR_t rider from the stored ids (cross-check against the tool's own eor.json / INDEX).

Usage: validate_traces.py RESULTS_DIR [--expect-windows 512] [--verify-hash] [--out FILE.json]
Exit 0 = every check passed; 1 = a check failed; 2 = unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_io as tio  # noqa: E402


def ir_mean(ids_l: np.ndarray, win: np.ndarray, phase_of_later: np.ndarray, want: int, E: int) -> tuple[float, int]:
    """mean |E_t n E_{t-1}|/K over adjacent rows of the same window whose later row has phase `want`."""
    a, b = ids_l[1:], ids_l[:-1]
    same = (win[1:] == win[:-1]) & (phase_of_later[1:] == want)
    if not same.any():
        return float("nan"), 0
    ma = np.zeros((int(same.sum()), E), bool)
    mb = np.zeros_like(ma)
    idx = np.arange(ma.shape[0])[:, None]
    ma[idx, a[same].astype(np.int64)] = True
    mb[idx, b[same].astype(np.int64)] = True
    return float((ma & mb).sum(1).mean() / ids_l.shape[1]), int(same.sum())


def validate(results_dir, expect_windows: int, verify_hash: bool, prefill_rows: int = 512) -> dict:
    found = tio.discover(results_dir)
    out: dict = {"results_dir": str(results_dir), "domains": {}, "problems": []}
    P = out["problems"]
    for dom, parts in sorted(found.items()):
        t = tio.load_domain(results_dir, dom, verify=verify_hash)
        d: dict = {"shards": {p: len(v) for p, v in parts.items()}, "rows": t.R, "layers": t.L,
                   "n_experts": sorted(set(t.n_experts.tolist())), "top_k": sorted(set(t.top_k.tolist()))}
        wins = np.unique(t.win)
        d["windows"] = int(wins.size)
        d["window_range"] = [int(wins.min()), int(wins.max())]
        if wins.size != expect_windows or not np.array_equal(wins, np.arange(expect_windows)):
            P.append(f"{dom}: windows are not exactly 0..{expect_windows - 1} ({wins.size} found)")
        # row accounting per window
        pf = np.bincount(t.win[t.phase == 0], minlength=expect_windows)
        dc = np.bincount(t.win[t.phase == 1], minlength=expect_windows)
        d["prefill_rows_per_window"] = sorted(set(pf.tolist()))
        d["decode_rows"] = int(dc.sum())
        d["decode_windows"] = int((dc > 0).sum())
        d["decode_rows_per_decode_window"] = [int(dc[dc > 0].min()), int(dc[dc > 0].max())] if (dc > 0).any() else None
        if (pf != prefill_rows).any():
            P.append(f"{dom}: {(pf != prefill_rows).sum()} windows without exactly {prefill_rows} prefill rows")
        # token order: pos strictly increasing inside a window; no prefill after decode
        same = t.win[1:] == t.win[:-1]
        if not (t.pos[1:][same] > t.pos[:-1][same]).all():
            P.append(f"{dom}: token order broken inside a window")
        if ((t.phase[1:] < t.phase[:-1]) & same).any():
            P.append(f"{dom}: prefill row after decode row")
        # no window split across a duplicate/overlap: a window must be one contiguous run
        runs = 1 + int((t.win[1:] != t.win[:-1]).sum())
        if runs != wins.size:
            P.append(f"{dom}: {runs} contiguous runs for {wins.size} windows (window repeated or interleaved)")
        # ids: in range, distinct per row, correct count
        irs = []
        for l in range(t.L):
            E, K = int(t.n_experts[l]), int(t.top_k[l])
            x = t.layer_ids(l)
            if x.min() < 0 or x.max() >= E:
                P.append(f"{dom} layer {l}: id outside 0..{E - 1}")
                continue
            s = np.sort(x, axis=1)
            if (s[:, 1:] == s[:, :-1]).any():
                P.append(f"{dom} layer {l}: duplicate expert inside one row")
            if l % 8 == 0 or l == t.L - 1:
                irs.append((l, ir_mean(x, t.win, t.phase, 0, E)[0], ir_mean(x, t.win, t.phase, 1, E)[0], K / E))
        d["ir_sample_layers"] = [{"layer": l, "ir_prefill": round(a, 6), "ir_decode": round(b, 6)} for l, a, b, _ in irs]
        # compare to the tool's own per-layer eor.json (shard-weighted) for the sampled layers
        diffs = []
        for part, dirs in parts.items():
            for sd in dirs:
                eor_p = Path(sd) / "routing_trace.eor.json"
                if not eor_p.exists():
                    P.append(f"{sd}: routing_trace.eor.json missing (no tool-side IR to cross-check)")
                    continue
                eor = json.loads(eor_p.read_text())
                tr = tio.load_shard(sd)
                for l, _, _, _ in irs:
                    E = int(tr.n_experts[l])
                    mine, n = ir_mean(tr.layer_ids(l), tr.win, tr.phase, 0, E)
                    theirs = eor["layers"][l]["prefill"]["ir_mean"]
                    diffs.append(abs(mine - theirs))
                    if n != eor["layers"][l]["prefill"]["pairs"]:
                        P.append(f"{sd} layer {l}: prefill pairs {n} != tool {eor['layers'][l]['prefill']['pairs']}")
        d["max_abs_ir_diff_vs_tool"] = float(max(diffs)) if diffs else None
        if diffs and max(diffs) > 2e-6:
            P.append(f"{dom}: IR recomputation differs from eor.json by {max(diffs):.2e}")
        out["domains"][dom] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--expect-windows", type=int, default=512)
    ap.add_argument("--prefill-rows", type=int, default=512)
    ap.add_argument("--verify-hash", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        r = validate(a.results_dir, a.expect_windows, a.verify_hash, a.prefill_rows)
    except (OSError, ValueError, KeyError) as e:
        print(f"unreadable: {e}", file=sys.stderr)
        return 2
    txt = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
    print(txt)
    print("PROBLEMS:" if r["problems"] else "ok: all checks passed", *r["problems"], sep="\n")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
