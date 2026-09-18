# Native Journal feed owner

`JournalFeedOwner` is the production read/ACK boundary for the bounded belief feed.
It consumes only the Journal frontier published after a successful event `fsync`;
newline-complete visible bytes are not sufficient.

The Journal stores a compact current publication record and one closed descriptor
seal per completed shard. These records are derived durability metadata, not a
second event log. They bind the source directory, ownership era, frontier, shard
device/inode/size/ctime/mode/link-count, and checksum. Rotation preserves the era
and seals the completed shard. Unknown, same-size, old-writer, or out-of-band changes
invalidate feed capability. `recover_durable_publication()` is the explicit operation
that streams, verifies and fsyncs old history before beginning a fresh ownership era.

The owner holds an exclusive per-reader lease, initially requires a private regular
single-link file owned by the effective user, and validates the lease path on every
operation. It returns bounded rows with an opaque HMAC token and accepts ACK only for
an exact returned `(seq, shard, byte offset, line)` position under the same source and
era. A later owner-published append does not invalidate an earlier token. The durable
cursor stores its exact position and shard identity, so ordinary restart does not scan
the acknowledged prefix. An outstanding batch may be replayed only with the exact
`DrainLimits` under which it was issued; changed bounds refuse until ACK.

Journal exclusion is thread-owned and genuinely same-thread reentrant. Other threads
on the same instance and distinct processes cannot bypass it. Lock admission is
nonblocking. Regular-file reads and `fsync` remain cooperative: Python cannot cancel
a kernel syscall already in progress. A resolved operation that returns after its
deadline is exposed as `JournalFeedOwner.last_overrun`; no hard `max_seconds` claim is
made, and slow corpus/Ledger/SQLite work remains outside Journal/control locks.

The belief feed uses the existing Vidya Ledger single-writer instance. It does not
write `ledger.py`, seed private cache fields, or claim concurrent publisher safety.
The feed can carry the separately frozen v2 observation-bound source schema, but this
owner introduces no scientific schema, effect, ranking, or nomination authority.
