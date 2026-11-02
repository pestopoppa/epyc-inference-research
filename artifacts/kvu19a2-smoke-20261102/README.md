# KVU-19a-2 model-free GPU capture smoke

One native FA case (`FA_CASE=78`, D=576/512, 8192 KV, F16, four query rows)
ran for each preserved baseline and candidate under an exclusive GPU and q2
build-role claim. Both exited 0 and reported zero NaNs. Same-process KFD
presence was observed for each; one live PID/birth sample per arm bound the
capture executable and four mapped ggml objects to their actual file digests
and device/inode identities. The two HIP digests match the prior build-pair
receipt. The capture executable hash matches its earlier compile-only
receipt. The two outputs have different SHA-256 digests.

This is one run per arm: it establishes neither within-arm nondeterminism nor
the cause or significance of the cross-arm difference. Its source tree was
mutable at build time, the other three ggml library hashes were computed from
the preserved copies at launch rather than independently bound to a source
build manifest, and GPU VRAM was not sampled. Both logs include a second
explicit backend-load warning; they also report a ROCm device and a completed
case. No numerical, promotion or production claim follows. The future repeated
run still requires byte-bound inputs, full source/build custody, in-window
residency, controls and belief wiring.

`result.json` records PIDs, start ticks, mapping samples and file hashes;
`producer.py`, logs and raw output bytes are retained alongside it. The
`kvu19a2_custody` helper separately supports SHA-256-checked sealed memfd
copies of source archive/patch bytes, and refuses incomplete or changed live
ggml mappings. A sealed archive does not make an extracted build tree or
toolchain immutable, and a sampling check does not prove compiler inputs.
