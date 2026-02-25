#!/bin/bash
set -euo pipefail

# Recover corpus DB from interrupted build:
#   1. Checkpoint WAL
#   2. Create ngram index
#   3. Resume remaining languages
#   4. Finalize
#
# Safe to re-run — each step is idempotent.

DB_DIR="/mnt/raid0/llm/cache/corpus/full_index"
DB="$DB_DIR/corpus.db"
LOG="$DB_DIR/recover.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build_index_v2.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Corpus recovery started ==="
log "DB: $DB ($(du -sh "$DB" 2>/dev/null | cut -f1))"
log "WAL: $(du -sh "$DB-wal" 2>/dev/null | cut -f1 || echo 'none')"

# Step 1: WAL checkpoint
if [ -f "$DB-wal" ] && [ "$(stat -c%s "$DB-wal" 2>/dev/null || echo 0)" -gt 1000 ]; then
  log "Step 1: Checkpointing WAL ($(du -sh "$DB-wal" | cut -f1))..."
  python3 -c "
import sqlite3, time
conn = sqlite3.connect('$DB', timeout=0)
conn.execute('PRAGMA busy_timeout = 0')
t0 = time.monotonic()
r = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
elapsed = time.monotonic() - t0
print(f'Checkpoint done in {elapsed:.1f}s: blocked={r[0]} written={r[1]} checkpointed={r[2]}')
conn.close()
" 2>&1 | tee -a "$LOG"
  log "Step 1: WAL checkpoint complete"
else
  log "Step 1: WAL already clean, skipping"
fi

# Step 2: Check DB health + create ngram index
log "Step 2: Checking DB state..."
python3 -c "
import sqlite3, time, json

conn = sqlite3.connect('$DB')
conn.execute('PRAGMA mmap_size=4294967296')  # 4GB mmap
conn.execute('PRAGMA cache_size=-512000')     # 512MB cache

# Count what we have
snippets = conn.execute('SELECT COUNT(*) FROM snippets').fetchone()[0]
ngrams = conn.execute('SELECT COUNT(*) FROM ngrams').fetchone()[0]
print(f'Snippets: {snippets:,}')
print(f'Ngrams:   {ngrams:,}')

langs = conn.execute('SELECT language, COUNT(*) FROM snippets GROUP BY language ORDER BY COUNT(*) DESC').fetchall()
for lang, cnt in langs:
    print(f'  {lang}: {cnt:,}')

# Check if ngram index exists
indexes = conn.execute(\"SELECT name FROM sqlite_master WHERE type='index'\").fetchall()
idx_names = [r[0] for r in indexes]
print(f'Indexes: {idx_names}')

if 'idx_ngrams_gram' not in idx_names:
    print('Creating idx_ngrams_gram index...')
    t0 = time.monotonic()
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram)')
    conn.commit()
    elapsed = time.monotonic() - t0
    print(f'Index created in {elapsed:.1f}s')
else:
    print('idx_ngrams_gram already exists')

conn.close()
" 2>&1 | tee -a "$LOG"
log "Step 2: Index creation complete"

# Step 3: Resume remaining languages
log "Step 3: Resuming build for remaining languages..."
python3 "$BUILD_SCRIPT" \
  --output "$DB_DIR" \
  --languages python,javascript,typescript,rust,go,c++ \
  --resume \
  --skip-finalize \
  2>&1 | tee -a "$LOG"
log "Step 3: Language builds complete"

# Step 4: Finalize (create index if new ngrams were added)
log "Step 4: Finalizing..."
python3 -c "
import sqlite3, time

conn = sqlite3.connect('$DB')
conn.execute('PRAGMA mmap_size=4294967296')
conn.execute('PRAGMA cache_size=-512000')

# Re-create index to cover new ngrams
print('Re-creating ngram index to cover new data...')
conn.execute('DROP INDEX IF EXISTS idx_ngrams_gram')
t0 = time.monotonic()
conn.execute('CREATE INDEX idx_ngrams_gram ON ngrams(gram)')
conn.commit()
elapsed = time.monotonic() - t0
print(f'Index rebuilt in {elapsed:.1f}s')

# WAL checkpoint to compact
print('Final WAL checkpoint...')
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')

snippets = conn.execute('SELECT COUNT(*) FROM snippets').fetchone()[0]
ngrams = conn.execute('SELECT COUNT(*) FROM ngrams').fetchone()[0]
langs = conn.execute('SELECT language, COUNT(*) FROM snippets GROUP BY language ORDER BY COUNT(*) DESC').fetchall()

print(f'Final state: {snippets:,} snippets, {ngrams:,} ngrams')
for lang, cnt in langs:
    print(f'  {lang}: {cnt:,}')

conn.close()
" 2>&1 | tee -a "$LOG"

log "=== Corpus recovery complete ==="
log "To use: update model_registry.yaml index_path to $DB_DIR"
