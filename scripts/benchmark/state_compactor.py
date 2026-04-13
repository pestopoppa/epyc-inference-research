#!/usr/bin/env python3
"""llama.cpp KV State Compactor for Attention Matching.

Reads a llama.cpp saved state file, removes selected positions,
optionally sets beta values on survivors, and writes a valid
compact state file that can be restored via /slots/{id}?action=restore.

Binary format (experimental llama.cpp with AM beta extension):
  File header: magic(u32) + version(u32) + n_tokens(u32) + tokens(i32[])
  KV state:    n_stream(u32), per-stream: cell_count(u32) + meta + data
  Meta/cell:   pos(i32) + n_seq_id(u32) + ext{x:i32, y:i32, beta:f32} + seq_ids(i32[])
  Data:        v_trans(u32) + n_layer(u32) + per-layer K + per-layer V

Usage:
    python state_compactor.py input.bin output.bin --keep-ratio 0.5
    python state_compactor.py input.bin output.bin --keep-positions 0,1,5,10,20
    python state_compactor.py input.bin --info  # just print state info
"""

import argparse
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import io as io_module

MAGIC_GGSQ = 0x67677371
VERSION = 2


@dataclass
class CellMeta:
    pos: int
    n_seq_id: int
    ext_x: int      # M-RoPE x
    ext_y: int      # M-RoPE y
    ext_beta: float  # AM attention bias
    seq_ids: List[int]


@dataclass
class LayerData:
    type_id: int
    row_size: int
    rows: List[bytes]  # one bytes object per cell


@dataclass
class StreamState:
    cell_count: int
    cells: List[CellMeta]
    v_trans: int
    n_layer: int
    k_layers: List[LayerData]
    v_layers: List[LayerData]


@dataclass
class SavedState:
    magic: int
    version: int
    n_tokens: int
    tokens: List[int]
    n_stream: int
    streams: List[StreamState]
    # opaque trailing bytes (e.g., recurrent state for SSM-hybrid models)
    # preserved unchanged through compaction
    tail_bytes: bytes = b""


def read_state(filepath: str) -> SavedState:
    """Parse a llama.cpp saved state binary."""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        assert magic == MAGIC_GGSQ, f"Bad magic: {magic:#x}"
        assert version == VERSION, f"Bad version: {version}"

        n_tokens = struct.unpack('<I', f.read(4))[0]
        tokens = list(struct.unpack(f'<{n_tokens}i', f.read(4 * n_tokens)))

        n_stream = struct.unpack('<I', f.read(4))[0]
        streams = []

        for s in range(n_stream):
            cell_count = struct.unpack('<I', f.read(4))[0]
            if cell_count == 0:
                streams.append(StreamState(0, [], 0, 0, [], []))
                continue

            # Read cell metadata
            cells = []
            for i in range(cell_count):
                pos = struct.unpack('<i', f.read(4))[0]
                n_seq_id = struct.unpack('<I', f.read(4))[0]
                # ext: {x: i32, y: i32, beta: f32}
                ext_x = struct.unpack('<i', f.read(4))[0]
                ext_y = struct.unpack('<i', f.read(4))[0]
                ext_beta = struct.unpack('<f', f.read(4))[0]
                seq_ids = list(struct.unpack(f'<{n_seq_id}i', f.read(4 * n_seq_id)))
                cells.append(CellMeta(pos, n_seq_id, ext_x, ext_y, ext_beta, seq_ids))

            # Read K/V data
            v_trans = struct.unpack('<I', f.read(4))[0]
            n_layer = struct.unpack('<I', f.read(4))[0]

            # K layers
            k_layers = []
            for il in range(n_layer):
                k_type = struct.unpack('<i', f.read(4))[0]
                k_row_size = struct.unpack('<Q', f.read(8))[0]
                rows = []
                for c in range(cell_count):
                    rows.append(f.read(k_row_size))
                k_layers.append(LayerData(k_type, k_row_size, rows))

            # V layers (only if !v_trans)
            v_layers = []
            if not v_trans:
                for il in range(n_layer):
                    v_type = struct.unpack('<i', f.read(4))[0]
                    v_row_size = struct.unpack('<Q', f.read(8))[0]
                    rows = []
                    for c in range(cell_count):
                        rows.append(f.read(v_row_size))
                    v_layers.append(LayerData(v_type, v_row_size, rows))

            streams.append(StreamState(cell_count, cells, v_trans, n_layer, k_layers, v_layers))

        # Read remaining bytes (recurrent state for SSM-hybrid models, or empty for pure attention)
        tail_bytes = f.read()

    return SavedState(magic, version, n_tokens, tokens, n_stream, streams, tail_bytes)


def write_state(state: SavedState, filepath: str):
    """Write a llama.cpp state binary."""
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', state.magic))
        f.write(struct.pack('<I', state.version))
        f.write(struct.pack('<I', state.n_tokens))
        f.write(struct.pack(f'<{state.n_tokens}i', *state.tokens))

        f.write(struct.pack('<I', state.n_stream))

        for stream in state.streams:
            f.write(struct.pack('<I', stream.cell_count))
            if stream.cell_count == 0:
                continue

            # Write cell metadata
            for cell in stream.cells:
                f.write(struct.pack('<i', cell.pos))
                f.write(struct.pack('<I', cell.n_seq_id))
                f.write(struct.pack('<i', cell.ext_x))
                f.write(struct.pack('<i', cell.ext_y))
                f.write(struct.pack('<f', cell.ext_beta))
                f.write(struct.pack(f'<{cell.n_seq_id}i', *cell.seq_ids))

            # Write K/V data
            f.write(struct.pack('<I', stream.v_trans))
            f.write(struct.pack('<I', stream.n_layer))

            for layer in stream.k_layers:
                f.write(struct.pack('<i', layer.type_id))
                f.write(struct.pack('<Q', layer.row_size))
                for row in layer.rows:
                    f.write(row)

            if not stream.v_trans:
                for layer in stream.v_layers:
                    f.write(struct.pack('<i', layer.type_id))
                    f.write(struct.pack('<Q', layer.row_size))
                    for row in layer.rows:
                        f.write(row)

        # Write trailing bytes (recurrent state for SSM-hybrid, preserved unchanged)
        if state.tail_bytes:
            f.write(state.tail_bytes)


def compact_state(
    state: SavedState,
    keep_indices: Optional[List[int]] = None,
    keep_ratio: float = 0.5,
    beta: float = 0.0,
    keep_first: int = 5,
    keep_last: int = 10,
) -> SavedState:
    """Compact a state by removing cells and optionally setting beta.

    If keep_indices is provided, keeps those cell indices.
    Otherwise uses a heuristic: keep first N, last M, sample middle.
    """
    new_streams = []

    for stream in state.streams:
        if stream.cell_count == 0:
            new_streams.append(stream)
            continue

        n = stream.cell_count

        if keep_indices is None:
            # Heuristic: keep first, last, evenly sampled middle
            keep = set()
            for i in range(min(keep_first, n)):
                keep.add(i)
            for i in range(max(0, n - keep_last), n):
                keep.add(i)
            # Fill to target ratio
            target = max(len(keep), int(n * keep_ratio))
            middle_start = min(keep_first, n)
            middle_end = max(0, n - keep_last)
            if middle_end > middle_start and target > len(keep):
                step = max(1, (middle_end - middle_start) // (target - len(keep)))
                for i in range(middle_start, middle_end, step):
                    keep.add(i)
                    if len(keep) >= target:
                        break
            keep_list = sorted(keep)
        else:
            keep_list = sorted(keep_indices)

        # Build compact stream
        new_cells = []
        new_k_layers = [LayerData(kl.type_id, kl.row_size, []) for kl in stream.k_layers]
        new_v_layers = [LayerData(vl.type_id, vl.row_size, []) for vl in stream.v_layers]

        for idx in keep_list:
            cell = stream.cells[idx]
            new_cell = CellMeta(
                pos=cell.pos,
                n_seq_id=cell.n_seq_id,
                ext_x=cell.ext_x,
                ext_y=cell.ext_y,
                ext_beta=beta if beta != 0.0 else cell.ext_beta,
                seq_ids=list(cell.seq_ids),
            )
            new_cells.append(new_cell)

            for il, kl in enumerate(stream.k_layers):
                new_k_layers[il].rows.append(kl.rows[idx])
            for il, vl in enumerate(stream.v_layers):
                new_v_layers[il].rows.append(vl.rows[idx])

        new_stream = StreamState(
            cell_count=len(keep_list),
            cells=new_cells,
            v_trans=stream.v_trans,
            n_layer=stream.n_layer,
            k_layers=new_k_layers,
            v_layers=new_v_layers,
        )
        new_streams.append(new_stream)

    return SavedState(
        magic=state.magic,
        version=state.version,
        n_tokens=state.n_tokens,
        tokens=list(state.tokens),
        n_stream=state.n_stream,
        streams=new_streams,
        tail_bytes=state.tail_bytes,  # preserve recurrent state unchanged
    )


def print_info(state: SavedState):
    """Print state info."""
    print(f"Magic: {state.magic:#x} ({'ggsq' if state.magic == MAGIC_GGSQ else 'UNKNOWN'})")
    print(f"Version: {state.version}")
    print(f"Tokens: {state.n_tokens}")
    print(f"Streams: {state.n_stream}")
    for si, stream in enumerate(state.streams):
        print(f"\n  Stream {si}: {stream.cell_count} cells")
        if stream.cell_count == 0:
            continue
        positions = [c.pos for c in stream.cells]
        betas = [c.ext_beta for c in stream.cells]
        nonzero_betas = sum(1 for b in betas if b != 0.0)
        print(f"    Positions: [{min(positions)}..{max(positions)}]")
        print(f"    Betas: {nonzero_betas} non-zero out of {len(betas)}")
        print(f"    v_trans: {stream.v_trans}")
        print(f"    Layers: {stream.n_layer}")
        if stream.k_layers:
            kl = stream.k_layers[0]
            print(f"    K type: {kl.type_id}, row_size: {kl.row_size} bytes")
        if stream.v_layers:
            vl = stream.v_layers[0]
            print(f"    V type: {vl.type_id}, row_size: {vl.row_size} bytes")
        total_kv = sum(kl.row_size * stream.cell_count for kl in stream.k_layers)
        total_kv += sum(vl.row_size * stream.cell_count for vl in stream.v_layers)
        print(f"    Total KV data: {total_kv / 1024:.1f} KB")
    if state.tail_bytes:
        print(f"\n  Tail bytes (recurrent/hybrid state): {len(state.tail_bytes)} bytes ({len(state.tail_bytes)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="llama.cpp KV State Compactor")
    parser.add_argument("input", help="Input state file")
    parser.add_argument("output", nargs="?", help="Output compacted state file")
    parser.add_argument("--info", action="store_true", help="Print state info and exit")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Fraction of cells to keep (default: 0.5)")
    parser.add_argument("--keep-positions", type=str, help="Comma-separated list of cell indices to keep")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta value to set on kept cells (0.0 = unchanged)")
    parser.add_argument("--keep-first", type=int, default=5, help="Always keep first N cells")
    parser.add_argument("--keep-last", type=int, default=10, help="Always keep last N cells")
    args = parser.parse_args()

    state = read_state(args.input)

    if args.info:
        print_info(state)
        return

    if not args.output:
        print("Error: output file required (unless --info)", file=sys.stderr)
        sys.exit(1)

    keep_indices = None
    if args.keep_positions:
        keep_indices = [int(x) for x in args.keep_positions.split(",")]

    compact = compact_state(
        state,
        keep_indices=keep_indices,
        keep_ratio=args.keep_ratio,
        beta=args.beta,
        keep_first=args.keep_first,
        keep_last=args.keep_last,
    )

    write_state(compact, args.output)

    orig_cells = sum(s.cell_count for s in state.streams)
    new_cells = sum(s.cell_count for s in compact.streams)
    ratio = orig_cells / max(new_cells, 1)
    print(f"Compacted: {orig_cells} → {new_cells} cells ({ratio:.1f}x compression)")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
