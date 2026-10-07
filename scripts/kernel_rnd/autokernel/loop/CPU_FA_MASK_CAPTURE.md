# DS41 real-mask correctness corpus

OP80 installs the real-mask check between the synthetic FA identity check and the
FA performance screen. The gate refuses with `oracle_unavailable` until the probe
supports captured masks, all sixteen masks/sidecars exist in `<store>/cpu_fa_real_masks`,
and the native run manifest agrees with every sidecar and byte digest.
Installing the C++ hook alone does not unblock candidates. No live lane inputs or
kernel anchors are changed by these preparations.

The files are `ds41_realmask_kv{4,8,32,64}k_nb{2,3,4,5}.mask.f16`: exactly
`kv * nb * 2` bytes, little-endian F16, token-major rows. Both probe arms receive
the same path through `--mask-file`. Missing, short and oversized files refuse;
the input digest includes the captured bytes. Q/K/V and sinks remain deterministic
probe inputs; this is representative mask-pattern coverage, not a replay of an
entire serving computation.

The experimental DS41 hook observes the **combined raw plus compressed top-k mask**
immediately before FLASH_ATTN_EXT. It refuses other head widths, multiple mask
streams, GPU residency and non-FA contexts. It captures only actual tensor widths
4096/8192/32768/65536 and query-row counts 2..5, taken from Q's token dimension
before its FA permutation. A padded mask keeps every original row in the graph;
the dump extracts precisely its first consumed N rows. Sidecars preserve the full
original dimensions/strides, row slice, source layer and compressed-group ratio.
The reader validates the single broadcast mask head/stream and contiguous slice.
DS41 has compressed groups with ratios two and one, so
an original prompt length of 4096 does not imply a 4096-column FA operand.
Never resize, pad, tile, truncate or relabel a mask to fill the corpus.

Prepare `capture-manifest.json` before starting the capture process, using
`python3 -m autokernel.loop.cpu_fa_mask_capture --source-root <experimental tree>
--build-dir <fresh reviewed build> --model <DS41 GGUF> --recipe-file <native launch JSON>
--prompt-file <actual fixed prompt> --capture-dir <new dir> --run-id <unique ID>`
under the narrow build claim. This hashes the original model, prompt, recipe and
build images, copies the prompt/recipe, and returns the exact launch environment.
No metadata is reconstructed when the oracle reads the files.

Each mask has an adjacent `.mask.json` recording its original UTC capture time,
actual dimensions, F16 layout, 64 query heads / one KV head, byte count and FNV1a64
byte digest, source commit, model path/model SHA256, prompt/recipe digests and run ID. First publication
uses an atomic exclusive link, so later layers cannot overwrite an earlier mask.
Use one capture process and one new directory per source/model/run. The mask
file and sidecar are separately published; inspect both after that process ends.

## Preparing a capture run

Do not run a capture until its prospective write-side claim carrier is wired
(VB-AK-REALMASK in root's adapter source table). The following is a preparation
recipe, not evidence of a completed capture or permission to alter a live lane.

1. Build the reviewed experimental hook tree into its own build directory with a
   narrow `region-lock run --cpu-list <owned cores> --role build` claim. Prove the
   binary/library mtime and source identity, inspect `strings` for
   `AUTOKERNEL_DUMP_FA_MASK`, and verify its own ggml linkage. Use absolute paths.
2. Launch a separate CPU-only DS41 server on an unused port under the same kind of
   claim, with its own `LD_LIBRARY_PATH`, FA enabled, no production process changes,
   and these nonempty variables set from the verified run inputs:

   ```text
   AUTOKERNEL_DUMP_FA_MASK=<new empty run directory>
   AUTOKERNEL_FA_MASK_SOURCE_COMMIT=<full hook-tree commit>
   AUTOKERNEL_FA_MASK_MODEL=<absolute model path>
   AUTOKERNEL_FA_MASK_MODEL_SHA256=<verified model digest>
   AUTOKERNEL_FA_MASK_RUN_ID=<unique run identifier>
   AUTOKERNEL_FA_MASK_RECIPE_SHA256=<original native recipe digest>
   AUTOKERNEL_FA_MASK_PROMPT_SHA256=<original prompt digest>
   ```

3. Evaluate an actual fixed prompt in that server with query batches of 2, 3, 4
   and 5 rows. Advance prompt depth until the combined FA width reaches each
   required value. Observe native sidecars, not context-token labels. The largest
   width may require an original context above 64k; use a separately validated
   context recipe, and retain the prompt, launch and context configuration in the
   prospective write-side carrier. A missing exact width remains unavailable.
4. Stop only that run's captured PID, confirm it is dead, and verify every sidecar
   and exact mask size/digest. Run anchor/candidate bit identity against that full
   directory using `check_real_mask_identity(..., capture_dir=<dir>)`, under a
   correctness build claim. Copy the reviewed corpus into the store only after
   its provenance and complete-file checks pass. This establishes no timing claim.

The regenerated `test-backend-ops-cpu-fa-longctx-v1.patch` registers the separate
synthetic correctness/performance corpus from `backend_ops_patch_block()`;
externally captured masks remain standalone-probe cases.
