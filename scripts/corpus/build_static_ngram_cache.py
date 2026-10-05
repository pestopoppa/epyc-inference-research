#!/usr/bin/env python3
"""Build a llama.cpp static n-gram lookup cache from bounded corpus chunks.

The native llama.cpp cache is token-id based. Build one cache per target model
tokenizer; do not share cache binaries across vocabularies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.registry.kernel_paths import backend_dir  # noqa: E402

DEFAULT_CORPUS_DIR = Path("/mnt/raid0/llm/cache/corpus/v3_sharded")
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
LARGE_SCAN_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ChunkRecord:
    path: str
    bytes: int
    snippets: int = 0


@dataclass
class BuildConfig:
    model: Path
    output: Path
    input_file: Path | None = None
    snippets_db: Path | None = None
    language: tuple[str, ...] = ()
    limit_snippets: int | None = None
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    max_bytes: int = DEFAULT_MAX_BYTES
    allow_large_scan: bool = False
    tmp_dir: Path | None = None
    llama_bin_dir: Path = field(default_factory=lambda: backend_dir("cpu"))
    threads: int | None = None
    dry_run: bool = False
    keep_parts: bool = False
    overwrite: bool = False
    manifest: Path | None = None


@dataclass
class BuildResult:
    schema_version: str
    created_at: str
    dry_run: bool
    model: str
    output: str
    manifest: str
    source: dict[str, object]
    llama_tools: dict[str, str]
    chunk_bytes: int
    max_bytes: int
    bytes_written: int
    chunks: list[ChunkRecord] = field(default_factory=list)
    part_paths: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["chunks"] = [asdict(chunk) for chunk in self.chunks]
        return data


CommandRunner = Callable[[Sequence[str]], None]


def _default_manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _tool_paths(llama_bin_dir: Path) -> dict[str, Path]:
    return {
        "lookup_create": llama_bin_dir / "llama-lookup-create",
        "lookup_merge": llama_bin_dir / "llama-lookup-merge",
    }


def _validate_config(config: BuildConfig) -> None:
    if config.input_file and config.snippets_db:
        raise ValueError("choose only one of input_file or snippets_db")
    if not config.input_file and not config.snippets_db:
        raise ValueError("input_file or snippets_db is required")
    if config.chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if config.max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    if config.max_bytes == 0 and not config.allow_large_scan:
        raise ValueError("--max-bytes 0 requires --allow-large-scan")
    if config.max_bytes > LARGE_SCAN_BYTES and not config.allow_large_scan:
        raise ValueError(
            f"max_bytes above {LARGE_SCAN_BYTES} requires --allow-large-scan"
        )
    if config.output.exists() and not config.overwrite and not config.dry_run:
        raise FileExistsError(f"output already exists: {config.output}")
    if not config.model.exists() and not config.dry_run:
        raise FileNotFoundError(f"model not found: {config.model}")
    if config.input_file and not config.input_file.exists():
        raise FileNotFoundError(f"input file not found: {config.input_file}")
    if config.snippets_db and not config.snippets_db.exists():
        raise FileNotFoundError(f"snippets DB not found: {config.snippets_db}")


def _run_subprocess(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _open_chunk(chunks_dir: Path, chunk_index: int) -> tuple[Path, object]:
    path = chunks_dir / f"chunk_{chunk_index:05d}.txt"
    return path, path.open("wb")


def _close_chunk(
    handle: object | None,
    path: Path | None,
    chunk_size: int,
    snippets: int,
    chunk_records: list[ChunkRecord],
) -> None:
    if handle is None or path is None:
        return
    handle.close()  # type: ignore[attr-defined]
    if chunk_size > 0:
        chunk_records.append(ChunkRecord(path=str(path), bytes=chunk_size, snippets=snippets))


def _bounded_payload(payload: bytes, remaining: int | None) -> bytes:
    if remaining is None:
        return payload
    return payload[:remaining]


def _chunk_input_file(source: Path, chunks_dir: Path, config: BuildConfig) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    max_remaining = None if config.max_bytes == 0 else config.max_bytes
    chunk_index = 0
    chunk_size = 0
    path: Path | None = None
    handle: object | None = None

    with source.open("rb") as reader:
        while True:
            if max_remaining is not None and max_remaining <= 0:
                break
            payload = reader.readline()
            if not payload:
                break
            payload = _bounded_payload(payload, max_remaining)
            offset = 0
            while offset < len(payload):
                if handle is None or path is None:
                    path, handle = _open_chunk(chunks_dir, chunk_index)
                room = config.chunk_bytes - chunk_size
                piece = payload[offset : offset + room]
                handle.write(piece)  # type: ignore[attr-defined]
                chunk_size += len(piece)
                offset += len(piece)
                if max_remaining is not None:
                    max_remaining -= len(piece)
                if chunk_size >= config.chunk_bytes:
                    _close_chunk(handle, path, chunk_size, 0, records)
                    chunk_index += 1
                    chunk_size = 0
                    path = None
                    handle = None

    _close_chunk(handle, path, chunk_size, 0, records)
    return records


def _iter_snippets(
    snippets_db: Path,
    *,
    languages: tuple[str, ...],
    limit: int | None,
) -> Iterable[tuple[int, str, str, str]]:
    where = ""
    params: list[object] = []
    if languages:
        placeholders = ",".join("?" for _ in languages)
        where = f"WHERE language IN ({placeholders})"
        params.extend(languages)
    limit_sql = ""
    if limit is not None:
        if limit < 0:
            raise ValueError("limit_snippets cannot be negative")
        limit_sql = "LIMIT ?"
        params.append(limit)

    query = f"SELECT id, code, source, language FROM snippets {where} ORDER BY id {limit_sql}"
    with sqlite3.connect(str(snippets_db)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query, params):
            yield int(row["id"]), str(row["code"]), str(row["source"]), str(row["language"])


def _chunk_snippets_db(
    snippets_db: Path,
    chunks_dir: Path,
    config: BuildConfig,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    max_remaining = None if config.max_bytes == 0 else config.max_bytes
    chunk_index = 0
    chunk_size = 0
    snippets = 0
    path: Path | None = None
    handle: object | None = None

    for snippet_id, code, source, language in _iter_snippets(
        snippets_db,
        languages=config.language,
        limit=config.limit_snippets,
    ):
        if max_remaining is not None and max_remaining <= 0:
            break
        header = f"\n\n// snippet_id={snippet_id} source={source} language={language}\n"
        payload = (header + code + "\n").encode("utf-8", errors="replace")
        payload = _bounded_payload(payload, max_remaining)
        if not payload:
            break

        offset = 0
        counted_chunk_index: int | None = None
        while offset < len(payload):
            if handle is None or path is None:
                path, handle = _open_chunk(chunks_dir, chunk_index)
            room = config.chunk_bytes - chunk_size
            piece = payload[offset : offset + room]
            if piece and counted_chunk_index != chunk_index:
                snippets += 1
                counted_chunk_index = chunk_index
            handle.write(piece)  # type: ignore[attr-defined]
            chunk_size += len(piece)
            offset += len(piece)
            if max_remaining is not None:
                max_remaining -= len(piece)
            if chunk_size >= config.chunk_bytes:
                _close_chunk(handle, path, chunk_size, snippets, records)
                chunk_index += 1
                chunk_size = 0
                snippets = 0
                path = None
                handle = None

    _close_chunk(handle, path, chunk_size, snippets, records)
    return records


def _lookup_create_command(
    *,
    create_bin: Path,
    model: Path,
    chunk: Path,
    part: Path,
    threads: int | None,
) -> list[str]:
    command = [
        str(create_bin),
        "-m",
        str(model),
        "-f",
        str(chunk),
        "-lcs",
        str(part),
    ]
    if threads is not None:
        command.extend(["-t", str(threads)])
    return command


def _lookup_merge_command(merge_bin: Path, parts: Sequence[Path], output: Path) -> list[str]:
    return [str(merge_bin), *[str(part) for part in parts], str(output)]


def build_static_ngram_cache(
    config: BuildConfig,
    *,
    runner: CommandRunner = _run_subprocess,
) -> BuildResult:
    _validate_config(config)
    tools = _tool_paths(config.llama_bin_dir)
    if not config.dry_run:
        for name, path in tools.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")

    manifest_path = config.manifest or _default_manifest_path(config.output)
    temp_parent = config.tmp_dir
    temp_context = (
        tempfile.TemporaryDirectory(prefix="static_ngram_cache_", dir=str(temp_parent))
        if temp_parent is not None
        else tempfile.TemporaryDirectory(prefix="static_ngram_cache_")
    )

    with temp_context as temp_name:
        work_dir = Path(temp_name)
        chunks_dir = work_dir / "chunks"
        parts_dir = work_dir / "parts"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)

        if config.input_file is not None:
            chunks = _chunk_input_file(config.input_file, chunks_dir, config)
            source: dict[str, object] = {"kind": "input_file", "path": str(config.input_file)}
        elif config.snippets_db is not None:
            chunks = _chunk_snippets_db(config.snippets_db, chunks_dir, config)
            source = {
                "kind": "snippets_db",
                "path": str(config.snippets_db),
                "language": list(config.language),
                "limit_snippets": config.limit_snippets,
            }
        else:  # pragma: no cover - guarded by validation
            raise AssertionError("missing source")

        if not chunks:
            raise RuntimeError("no source text selected for cache build")

        parts = [parts_dir / f"part_{idx:05d}.bin" for idx, _ in enumerate(chunks)]
        commands: list[list[str]] = []
        for chunk, part in zip(chunks, parts, strict=True):
            command = _lookup_create_command(
                create_bin=tools["lookup_create"],
                model=config.model,
                chunk=Path(chunk.path),
                part=part,
                threads=config.threads,
            )
            commands.append(command)
            if not config.dry_run:
                runner(command)

        merge_command = _lookup_merge_command(tools["lookup_merge"], parts, config.output)
        commands.append(merge_command)
        if not config.dry_run:
            config.output.parent.mkdir(parents=True, exist_ok=True)
            if config.output.exists() and config.overwrite:
                config.output.unlink()
            runner(merge_command)
            if config.keep_parts:
                keep_dir = config.output.with_suffix(config.output.suffix + ".parts")
                keep_dir.mkdir(parents=True, exist_ok=True)
                for part in parts:
                    if part.exists():
                        shutil.copy2(part, keep_dir / part.name)

        result = BuildResult(
            schema_version="static_ngram_cache_build.v1",
            created_at=datetime.now(UTC).isoformat(),
            dry_run=config.dry_run,
            model=str(config.model),
            output=str(config.output),
            manifest=str(manifest_path),
            source=source,
            llama_tools={name: str(path) for name, path in tools.items()},
            chunk_bytes=config.chunk_bytes,
            max_bytes=config.max_bytes,
            bytes_written=sum(chunk.bytes for chunk in chunks),
            chunks=chunks,
            part_paths=[str(part) for part in parts],
            commands=commands,
            notes=[
                "Cache token ids are vocab-locked to the target model tokenizer.",
                "Use --lookup-cache-static/-lcs with the same target model family for A/B only.",
            ],
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        return result


def _parse_bytes(value: str) -> int:
    value = value.strip().lower()
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a static llama.cpp n-gram cache from bounded corpus chunks."
    )
    parser.add_argument("--model", required=True, type=Path, help="Target GGUF model/tokenizer.")
    parser.add_argument("--output", required=True, type=Path, help="Output static cache .bin path.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-file", type=Path, help="Plain text/code source file.")
    source.add_argument("--snippets-db", type=Path, help="v3 corpus snippets.db source.")
    parser.add_argument("--language", action="append", default=[], help="Repeatable snippet language filter.")
    parser.add_argument("--limit-snippets", type=int, help="Maximum snippets to read from snippets.db.")
    parser.add_argument(
        "--chunk-bytes",
        type=_parse_bytes,
        default=DEFAULT_CHUNK_BYTES,
        help="Maximum bytes per lookup-create input chunk.",
    )
    parser.add_argument(
        "--max-bytes",
        type=_parse_bytes,
        default=DEFAULT_MAX_BYTES,
        help="Maximum source bytes to scan; 0 means unbounded and requires --allow-large-scan.",
    )
    parser.add_argument(
        "--allow-large-scan",
        action="store_true",
        help="Allow scans above the safety cap or unbounded scans.",
    )
    parser.add_argument("--tmp-dir", type=Path, help="Parent temp directory for chunks and parts.")
    parser.add_argument("--llama-bin-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, help="Threads passed to llama-lookup-create.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/chunks, skip tool execution.")
    parser.add_argument("--keep-parts", action="store_true", help="Copy part caches next to output.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output.")
    parser.add_argument("--manifest", type=Path, help="Manifest JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = BuildConfig(
        model=args.model,
        output=args.output,
        input_file=args.input_file,
        snippets_db=args.snippets_db
        if args.input_file is not None or args.snippets_db is not None
        else DEFAULT_CORPUS_DIR / "snippets.db",
        language=tuple(args.language),
        limit_snippets=args.limit_snippets,
        chunk_bytes=args.chunk_bytes,
        max_bytes=args.max_bytes,
        allow_large_scan=args.allow_large_scan,
        tmp_dir=args.tmp_dir,
        llama_bin_dir=args.llama_bin_dir if args.llama_bin_dir is not None else backend_dir("cpu"),
        threads=args.threads,
        dry_run=args.dry_run,
        keep_parts=args.keep_parts,
        overwrite=args.overwrite,
        manifest=args.manifest,
    )
    try:
        result = build_static_ngram_cache(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
