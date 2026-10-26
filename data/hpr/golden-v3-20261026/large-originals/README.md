# HPR golden v3 large original byte archives

Lossless custody copies of four native captures omitted from research commit `3f2ead26abf2b44898b4c2cf2bb7eed970bdb98c` because each original exceeded 5 MB. The existing 44 native evidence files remain unchanged.

| Original relative path | Lossless archive | Original bytes | Original SHA-256 |
|---|---|---:|---|
| `64k-coarse-native/divergent_edit.raw.json` | [gzip](64k-coarse-native/divergent_edit.raw.json.gz) | 5392996 | `bf2241b9840850bfd14c4c1da03001266ac1241eef6f9b292d0e771440c96261` |
| `64k-coarse-native/matched_extension.raw.json` | [gzip](64k-coarse-native/matched_extension.raw.json.gz) | 5392910 | `3568a5a229ecfb4ea1e1798a42c0a2b23dc3ceee64b09ebc8385c85618325785` |
| `64k-dense-native/divergent_edit.raw.json` | [gzip](64k-dense-native/divergent_edit.raw.json.gz) | 5393020 | `9fa92cee639be356aa3ff4d4ae3c56d6068172d90179a22b23a9ceb2e106bba6` |
| `64k-dense-native/matched_extension.raw.json` | [gzip](64k-dense-native/matched_extension.raw.json.gz) | 5392901 | `17929e7663a569a3741938de5876a076aace4a4e06caf93f13c1de31e2b86f5b` |

The run-generated [archive manifest](archive-manifest.json) records pre/post original and complete decompressed byte counts and SHA-256, plus compressed byte counts and SHA-256 and actual archive timestamps. Compression uses `gzip -n -c`; verification consumes the entire `gzip -dc` stream. Verify compressed hashes first, then full decompressed byte counts and SHA-256 against this table and the native audit manifest. The originals remain at the source tmp path; no source file is rewritten.

This preserves accepted evidence bytes. The accepted result remains 24 fresh captures / 12 comparisons. The 72 reader projections are not 72 probes. Native contract remains UNVERIFIED; no historical regrade, performance claim, parent HPR closure, measurement warrant, or new measurement class follows from archival.
