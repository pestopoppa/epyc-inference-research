"""Prepare operator-reviewable community hypotheses; never inject or start runs.

Input is a JSON object with a ``claims`` list of human-extracted source claims.
The operator supplies a current hotspot export (a JSON list of Hotspot.to_dict
rows) and the live experiment database. Sources, applicability and units are
explicit; missing bandwidth is reported as unknown rather than estimated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from urllib.parse import urlsplit


class IntakeRefused(ValueError):
    pass


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or '\n' in value:
        raise IntakeRefused(f'{key} must be nonempty single-line text')
    return value.strip()


def prepare(claims, hotspots, db_path):
    """Return a Markdown review bundle. Database is opened read-only."""
    if not isinstance(claims, list) or not isinstance(hotspots, list):
        raise IntakeRefused('claims and hotspots must be lists')
    db_path = Path(db_path).resolve(strict=True)
    uri = db_path.as_uri() + '?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute('BEGIN')  # One consistent read snapshot across the sweep.
        lines = ['# Community intake — operator review required', '',
                 'Draft only. Review provenance, scope and nulls before manual inbox injection.', '']
        seen = set()
        for row in claims:
            if not isinstance(row, dict):
                raise IntakeRefused('claim must be an object')
            name, mechanism, symbol = (_text(row, k) for k in ('title', 'mechanism_id', 'target_symbol'))
            statement, falsifier, surface = (_text(row, k) for k in ('statement', 'falsifier', 'target_surface'))
            url, tier = _text(row, 'url'), _text(row, 'tier')
            if tier not in ('upstream', 'community', 'paper') or urlsplit(url).scheme not in ('http', 'https') or not urlsplit(url).netloc:
                raise IntakeRefused('tier or provenance URL invalid')
            key = (url, mechanism)
            if key in seen:
                raise IntakeRefused('duplicate source and mechanism')
            seen.add(key)
            achieved, ceiling = row.get('achieved_gb_s'), row.get('ceiling_gb_s')
            if achieved is None and ceiling is None:
                bw = 'unknown (source did not report comparable bandwidth and ceiling)'
            elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (achieved, ceiling)):
                raise IntakeRefused('bandwidth and ceiling must both be positive finite GB/s')
            else:
                basis = _text(row, 'bandwidth_basis')
                ratio = achieved / ceiling
                if not math.isfinite(ratio):
                    raise IntakeRefused('bandwidth ratio is not finite')
                bw = f'{ratio:.1%} ({achieved:g} / {ceiling:g} GB/s; {basis})'
                if achieved > ceiling:
                    bw += '; above ceiling: inspect cache residency and denominators'
            matches = [h for h in hotspots if isinstance(h, dict) and symbol.casefold() in str(h.get('signature', '')).casefold()]
            profile = ', '.join(f"{h['signature']} ({h.get('share_of_device_time', 'unknown')} device share)" for h in matches) or 'no matching signature; applicability unproven'
            nulls = conn.execute('SELECT attempt_id, epoch_sha256, target_surface, target_symbol, mechanism_id FROM experiments WHERE status = ? AND (mechanism_id = ? OR target_symbol = ?) ORDER BY recorded_at', ('measured_null', mechanism, symbol)).fetchall()
            history = '; '.join(' / '.join(str(x or '?') for x in result) for result in nulls) or 'no exact mechanism or symbol match; related nulls may exist'
            lines.extend([f'## {mechanism} (proposed) — {name}', f'Source ({tier}): {url}',
                          f'**Question:** {statement}', f'**Target:** {surface} / {symbol}', f'**Falsifier:** {falsifier}',
                          f'Bandwidth utilization: {bw}', f'Hotspot mapping: {profile}',
                          f'Known measured nulls (attempt / epoch / surface / symbol / mechanism): {history}',
                          'Upstream changes are gfx90a port hypotheses, not merge authorization.' if tier == 'upstream' else 'Source is a design prior, not a local measurement.', ''])
        return '\n'.join(lines)
    finally:
        conn.close()



def upstream_candidates(repo, ref):
    """Enumerate upstream CUDA/HIP file changes since the frozen fork, for human triage."""
    tip = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify',
                          '--end-of-options', f'{ref}^{{commit}}'], check=True,
                         capture_output=True, text=True, timeout=30).stdout.strip()
    command = ['git', '-C', str(repo), 'log', '--reverse', '--max-count=10001',
               '--format=%H%x09%s', f'0db32c06..{tip}', '--',
               'ggml/src/ggml-cuda', 'ggml/src/ggml-hip']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    rows = result.stdout.splitlines()
    if len(rows) > 10000:
        raise IntakeRefused('upstream commit list exceeds bound; narrow the sweep explicitly')
    return tip, ['- https://github.com/ggml-org/llama.cpp/commit/' + row.replace('\t', ' — ', 1)
                 for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--hotspots', type=Path, required=True)
    parser.add_argument('--experiments-db', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--upstream-repo', type=Path)
    parser.add_argument('--upstream-ref')
    args = parser.parse_args(argv)
    if (args.upstream_repo is None) != (args.upstream_ref is None):
        parser.error('upstream repo and ref must be supplied together')
    output_path = args.out.resolve()
    if ('inbox' in output_path.parts
            or output_path.is_relative_to(args.experiments_db.resolve().parent)):
        parser.error('output must be a review bundle outside the experiment store and inbox')
    claims_bytes, profile_bytes = args.claims.read_bytes(), args.hotspots.read_bytes()
    result = prepare(json.loads(claims_bytes)['claims'], json.loads(profile_bytes), args.experiments_db)
    if args.upstream_repo is not None:
        tip, candidates = upstream_candidates(args.upstream_repo, args.upstream_ref)
        result += (f'\n## Upstream CUDA/HIP commits to triage (tip {tip})\n'
                   + '\n'.join(candidates) + '\n')
    provenance = (f"Inputs: claims sha256 {hashlib.sha256(claims_bytes).hexdigest()}, "
                  f"hotspots sha256 {hashlib.sha256(profile_bytes).hexdigest()}, "
                  f"experiments database {args.experiments_db.resolve()}.\n\n")
    with args.out.open('x', encoding='utf-8') as output:
        output.write(provenance + result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
