"""Process-parallel KB-RAG re-embed driver (OP-24 [Q]/[D] migration).

Why this exists
---------------
`kb_rag.build_index()` encodes chunks serially in one process, and
`colbert_encoder` bounds ONNX to a handful of intra-op threads because the
request path issues single-row forward passes. That pairing is correct online
and is ~4.6% of a 192-thread host offline: a 28k-chunk re-embed takes hours at
~9 busy threads.

The work is embarrassingly parallel — one independent single-row forward pass
per chunk — so this driver fans out at the PROCESS level:

  phase 1  chunk the corpus (parallel over files) -> a full chunk manifest
  phase 2  encode every MISSING embedding in N worker processes, each with
           COLBERT_ENCODE_ONNX_THREADS=1..2. Workers only write per-chunk .npz
           files, which need no coordination.
  phase 3  ONE process rebuilds the sqlite catalog + FTS from the manifest.

SQLite is the single contention point and never sees more than one writer.

Resumability / idempotence
--------------------------
Phase 2 skips any chunk whose .npz already exists and loads, and every write is
atomic (tmp file + os.replace), so a kill at any instant leaves the store in a
resumable state: re-running re-does only what is genuinely missing. Phase 3 is a
full rebuild of catalog metadata from the manifest, so a catalog truncated by a
crash self-heals.

Worker lifetime
---------------
Workers are children of this process, are tracked by captured PID, and arm
PR_SET_PDEATHSIG(SIGKILL) so they cannot outlive the driver. No process is ever
selected by name pattern.

Usage:
  python3 scripts/kb_rag/parallel_reembed.py pilot --configs 48x1,96x1,160x1,88x2
  python3 scripts/kb_rag/parallel_reembed.py run --workers 160 --onnx-threads 1
  python3 scripts/kb_rag/parallel_reembed.py catalog
  python3 scripts/kb_rag/parallel_reembed.py verify
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

# MUST precede the numpy import: numpy's BLAS backend sizes its thread pool from
# these at first use, defaulting to one thread per visible core. Parallelism here
# is across PROCESSES, one chunk each, so per-worker pools are pure overhead.
#
# KNOWN-INCOMPLETE, measured 2026-08-12 on this host and stated so rather than
# assumed fixed: setting these does NOT drop the per-worker thread count. Each
# worker still carries 1 + len(affinity mask) threads (177 under a 176-cpu mask,
# 5 under a 4-cpu mask, 16 under a 16-cpu mask), and aggregate throughput was
# unchanged (61 -> 60 chunk/s). Bisecting the imports puts the pool behind the
# tokenizer, not ONNX Runtime (intra_op=1 verified) and not BLAS: a bare
# session.run() adds zero threads, tokenizer.encode() adds mask-size-minus-one.
# RAYON_NUM_THREADS=1 does not suppress it either.
#
# The fix, when someone picks this up: pin each worker to ONE distinct cpu with
# os.sched_setaffinity in the initializer, which bounds the pool by construction
# instead of by env var. Expected gain is large — 48 workers sustained ~185
# chunk/s while 176 workers sustained ~60, i.e. this driver is currently ~3x off
# its own best measured point and still finished 17k chunks in ~6 minutes.
for _var in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from src.retrieval import colbert_encoder  # noqa: E402
from src.retrieval.markdown_chunker import chunk_file  # noqa: E402

DEFAULT_CONFIG = _REPO / "config" / "kb_rag_config.yaml"

# Logical CPUs the workers may use. 184-191 is the declared GPU host lane and
# 88-95 are its SMT siblings (same physical cores) — both are reserved so a
# concurrent GPU benchmark keeps uncontended host threads.
GPU_HOST_LANE = set(range(184, 192)) | set(range(88, 96))
WORKER_CPUS = sorted(set(range(os.cpu_count() or 192)) - GPU_HOST_LANE)


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this process when its parent dies."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 — best-effort hardening
        pass


# ── phase 1: manifest ────────────────────────────────────────────────────────


def _chunk_one(args):
    path, max_chars = args
    try:
        chunks = chunk_file(path, max_chars=max_chars)
        mtime = Path(path).stat().st_mtime
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": str(e), "chunks": []}
    return {
        "path": path,
        "mtime": mtime,
        "chunks": [
            {
                "heading_path": c.heading_path,
                "line_start": c.line_range[0],
                "line_end": c.line_range[1],
                "content_hash": c.content_hash,
                "text": c.text,
            }
            for c in chunks
        ],
    }


def build_manifest(config_path: Path, jobs: int) -> list[dict]:
    from src.retrieval.kb_rag import CorpusConfig, _emb_relative_path, _walk_corpus

    cfg = CorpusConfig.from_yaml(config_path)
    files = _walk_corpus(cfg)
    with mp.Pool(processes=jobs, initializer=_set_pdeathsig) as pool:
        results = pool.map(_chunk_one, [(str(f), cfg.max_chunk_chars) for f in files], chunksize=8)

    manifest: list[dict] = []
    for r in results:
        if r.get("error"):
            print(f"  chunker failed on {r['path']}: {r['error']}", file=sys.stderr)
            continue
        for c in r["chunks"]:
            manifest.append(
                {
                    "file_path": r["path"],
                    "mtime": r["mtime"],
                    "emb_rel": _emb_relative_path(r["path"], c["content_hash"]),
                    **c,
                }
            )
    return manifest


# ── phase 2: parallel encode ─────────────────────────────────────────────────

_WORKER_STATE: dict = {}


@colbert_encoder.state_transaction
def _parallel_contract(index_dir: Path, *, load_encoder: bool = False) -> tuple[int, str]:
    """Resolve cap and role together without mutating catalog identity."""
    import sqlite3
    from contextlib import closing

    from src.retrieval import colbert_encoder
    from src.retrieval.kb_rag import (
        _index_convention, _index_token_caps, _read_index_meta,
        _roles_for_convention, _warn_on_encoder_drift,
    )

    meta = _read_index_meta(index_dir)  # reject bad metadata before model loading
    catalog = index_dir / "catalog.sqlite"
    convention = None
    if catalog.exists():
        with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            convention = _index_convention(conn)
    if convention is None and meta:
        convention = colbert_encoder.LEGACY_CONVENTION
    if convention is None:
        doc_cap = colbert_encoder.max_document_tokens()
        convention = colbert_encoder.PREFIX_CONVENTION
    else:
        _, doc_cap = _index_token_caps(meta)
    _, role = _roles_for_convention(convention)
    if load_encoder:
        if not colbert_encoder.ensure_loaded():
            raise RuntimeError("colbert encoder failed to load")
        _warn_on_encoder_drift(meta)
    return doc_cap, role


@colbert_encoder.state_transaction
def _prepare_encoding_contract(index_dir: Path) -> tuple[int, str]:
    """Stamp only a genuinely new output before any worker/manifest work."""
    from src.retrieval.kb_rag import (
        _ensure_catalog, _index_convention, _read_meta, _writer_convention,
    )

    contract = _parallel_contract(index_dir, load_encoder=True)
    conn = _ensure_catalog(index_dir, verify_tokenizer=True)
    try:
        if (
            _index_convention(conn) is None and not _read_meta(conn)
            and any((index_dir / "emb").rglob("*.npz"))
        ):
            raise RuntimeError("cannot stamp new caps over embeddings without catalog identity")
        _writer_convention(conn)
        conn.commit()
    finally:
        conn.close()
    return contract


def _encode_init(index_dir: str, onnx_threads: int, cpus: list[int]) -> None:
    _set_pdeathsig()
    os.environ["COLBERT_ENCODE_ONNX_THREADS"] = str(onnx_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(onnx_threads))
    if cpus:
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass
    from src.retrieval import colbert_encoder
    with colbert_encoder.locked_state():
        doc_cap, role = _parallel_contract(Path(index_dir), load_encoder=True)
        _WORKER_STATE["enc"] = colbert_encoder
        _WORKER_STATE["index_dir"] = Path(index_dir)
        _WORKER_STATE["max_tokens"] = doc_cap
        _WORKER_STATE["role"] = role
        _WORKER_STATE["generation"] = colbert_encoder.generation()


def _encode_task(task: tuple[str, str, str]) -> tuple[str, int, str]:
    """(emb_rel, text, role) -> (emb_rel, n_tokens, status)."""
    emb_rel, text, role = task
    if role != _WORKER_STATE["role"]:
        raise RuntimeError("encode task role differs from stored index convention")
    enc = _WORKER_STATE["enc"]
    with enc.locked_state():
        if _WORKER_STATE["generation"] != enc.generation():
            raise RuntimeError("worker encoder generation changed; reinitialize worker")
        out = _WORKER_STATE["index_dir"] / emb_rel
        if out.exists():
            try:
                with np.load(out) as z:
                    return emb_rel, int(z["emb"].shape[0]), "skipped"
            except Exception:  # noqa: BLE001 — truncated by a crash; re-encode
                pass
        emb = enc.encode(text, _WORKER_STATE["max_tokens"], role=role)
        if emb is None:
            return emb_rel, 0, "failed"
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".npz.tmp")
        try:
            # Write through the open fd: np.savez_compressed would otherwise append
            # ".npz" to a path that does not already end in it, and the atomic
            # rename would then move a file that does not exist.
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(fh, emb=emb)
            os.replace(tmp, out)
        except Exception as e:  # noqa: BLE001
            for leftover in (tmp, tmp + ".npz"):
                try:
                    os.unlink(leftover)
                except OSError:
                    pass
            return emb_rel, 0, f"error:{e}"
        return emb_rel, int(emb.shape[0]), "encoded"


def encode_missing(
    tasks: list[tuple[str, str, str]],
    index_dir: Path,
    workers: int,
    onnx_threads: int,
    label: str,
) -> dict:
    """Fan `tasks` out over `workers` processes. Returns timing + counts."""
    if not tasks:
        return {"tasks": 0, "encoded": 0, "elapsed_sec": 0.0, "chunks_per_sec": 0.0}
    _, role = _prepare_encoding_contract(index_dir)
    if any(task[2] != role for task in tasks):
        raise RuntimeError("encode tasks differ from stored index convention")
    started = time.perf_counter()
    counts = {"encoded": 0, "skipped": 0, "failed": 0, "error": 0}
    pids: list[int] = []
    pool = mp.Pool(
        processes=workers,
        initializer=_encode_init,
        initargs=(str(index_dir), onnx_threads, WORKER_CPUS),
    )
    try:
        pids = [p.pid for p in pool._pool]  # captured PIDs — never a name pattern
        (index_dir / ".worker_pids").write_text("\n".join(str(p) for p in pids) + "\n")
        done = 0
        for _rel, _n, status in pool.imap_unordered(_encode_task, tasks, chunksize=4):
            key = status if status in counts else "error"
            counts[key] += 1
            done += 1
            if done % 500 == 0:
                rate = done / (time.perf_counter() - started)
                print(f"  [{label}] {done}/{len(tasks)} {rate:.1f} chunk/s", flush=True)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
        pool.join()
    # Verify every captured PID is gone.
    alive = [p for p in pids if _pid_alive(p)]
    for p in alive:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    still = [p for p in pids if _pid_alive(p)]
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "workers": workers,
        "onnx_threads": onnx_threads,
        "tasks": len(tasks),
        **counts,
        "elapsed_sec": round(elapsed, 2),
        "chunks_per_sec": round(len(tasks) / elapsed, 2),
        "worker_pids": pids,
        "worker_pids_still_alive": still,
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ── phase 3: single-writer catalog rebuild ───────────────────────────────────


@colbert_encoder.state_transaction
def rebuild_catalog(manifest: list[dict], index_dir: Path) -> dict:
    import sqlite3

    from src.retrieval.kb_rag import (
        _ensure_catalog,
        _ensure_fts,
        _sync_chunk_fts_row,
    )

    _prepare_encoding_contract(index_dir)
    conn = _ensure_catalog(index_dir, verify_tokenizer=True)
    conn.row_factory = sqlite3.Row
    fts_enabled = _ensure_fts(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunk")
    if fts_enabled:
        cur.execute("DELETE FROM chunk_fts")

    inserted = 0
    missing_emb = 0
    for rec in manifest:
        emb_abs = index_dir / rec["emb_rel"]
        if not emb_abs.exists():
            missing_emb += 1
            continue
        try:
            with np.load(emb_abs) as z:
                n_tokens = int(z["emb"].shape[0])
        except Exception:  # noqa: BLE001
            missing_emb += 1
            continue
        cur.execute(
            "INSERT INTO chunk "
            "(file_path, heading_path, line_start, line_end, content_hash, "
            " mtime, emb_path, text_preview, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec["file_path"],
                json.dumps(rec["heading_path"]),
                rec["line_start"],
                rec["line_end"],
                rec["content_hash"],
                rec["mtime"],
                rec["emb_rel"],
                rec["text"].strip()[:240],
                n_tokens,
            ),
        )
        _sync_chunk_fts_row(
            cur, int(cur.lastrowid), rec["file_path"], rec["heading_path"],
            rec["text"], fts_enabled,
        )
        inserted += 1
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return {"catalog_rows": inserted, "manifest_records": len(manifest),
            "missing_embeddings": missing_emb, "fts": fts_enabled}


# ── orchestration ────────────────────────────────────────────────────────────


def _tasks_from_manifest(manifest: list[dict], index_dir: Path, role: str) -> list[tuple]:
    """Deduplicate to one encode task per distinct .npz still missing."""
    seen: set[str] = set()
    tasks: list[tuple[str, str, str]] = []
    for rec in manifest:
        rel = rec["emb_rel"]
        if rel in seen:
            continue
        seen.add(rel)
        if (index_dir / rel).exists():
            continue
        tasks.append((rel, rec["text"], role))
    return tasks


def _load_or_build_manifest(args) -> list[dict]:
    cache = Path(args.manifest)
    if cache.exists() and not args.rechunk:
        return json.loads(cache.read_text())
    print("phase 1: chunking corpus ...", flush=True)
    t0 = time.perf_counter()
    manifest = build_manifest(Path(args.config), jobs=min(64, len(WORKER_CPUS)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(manifest))
    print(f"  {len(manifest)} chunks from corpus in {time.perf_counter()-t0:.1f}s", flush=True)
    return manifest


def main() -> int:
    from src.retrieval import colbert_encoder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["pilot", "run", "catalog", "verify"])
    ap.add_argument("--index-dir", default=str(_REPO / "data" / "kb_rag" / "index-qd-v1"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default=str(_REPO / "data" / "kb_rag" / ".manifest-qd-v1.json"))
    ap.add_argument("--rechunk", action="store_true")
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--onnx-threads", type=int, default=1)
    ap.add_argument("--configs", default="48x1,96x1,160x1,88x2")
    ap.add_argument("--pilot-chunks", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    role = colbert_encoder.ROLE_DOCUMENT
    print(f"worker cpu set: {len(WORKER_CPUS)} logical cpus "
          f"(reserved for GPU host lane: {sorted(GPU_HOST_LANE)})", flush=True)

    if args.mode == "verify":
        manifest = _load_or_build_manifest(args)
        missing = _tasks_from_manifest(manifest, index_dir, role)
        print(json.dumps({
            "manifest_chunks": len(manifest),
            "distinct_embeddings_expected": len({r["emb_rel"] for r in manifest}),
            "embeddings_missing": len(missing),
            "npz_on_disk": len(list((index_dir / "emb").glob("*.npz"))),
        }, indent=2))
        return 0

    _, role = _prepare_encoding_contract(index_dir)
    if args.mode == "catalog":
        manifest = _load_or_build_manifest(args)
        print(json.dumps(rebuild_catalog(manifest, index_dir), indent=2))
        return 0

    manifest = _load_or_build_manifest(args)
    tasks = _tasks_from_manifest(manifest, index_dir, role)
    print(f"phase 2: {len(tasks)} embeddings missing of "
          f"{len({r['emb_rel'] for r in manifest})} expected", flush=True)

    if args.mode == "pilot":
        results = []
        offset = 0
        for spec in args.configs.split(","):
            w, t = spec.split("x")
            slice_ = tasks[offset:offset + args.pilot_chunks]
            offset += args.pilot_chunks
            if not slice_:
                break
            res = encode_missing(slice_, index_dir, int(w), int(t), spec)
            print(json.dumps(res), flush=True)
            results.append(res)
        best = max(results, key=lambda r: r["chunks_per_sec"])
        print(json.dumps({"pilot": results, "best": best["label"]}, indent=2))
        return 0

    if args.limit:
        tasks = tasks[: args.limit]
    res = encode_missing(tasks, index_dir, args.workers, args.onnx_threads, "full")
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
