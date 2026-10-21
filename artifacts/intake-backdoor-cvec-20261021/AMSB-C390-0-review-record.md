# AMSB-C390-0 — SEC-ART-3-shape review record: nerkyor Coder390-EfficientThink (Qwen3.8-27B post-train)

Date 2026-10-21. Reviewer: intake-backdoor-cvec subagent (no-compute). Intake: intake-2193 (primary), intake-2194 (relay, not corroboration).
Scope: HF metadata, small text files, and **GGUF header prefixes only (HTTP Range, 16 MiB, never tensor data)**. No weights downloaded, no model loaded.

## 0. Verdict

**GO for the Q8_0 download, conditional, for a tool-free same-harness A/B only (AMSB-C390-1). NO-GO for any tool access, registry admission, or role binding** (SEC-ART-3 would refuse tool access on this record). The review found **no integrity defect** (no sidecars, no code, no adapters, template and tokenizer clean) but **provenance cannot be established**: no training data is published, the predecessor was refusal-ablated, and teacher-output licensing is unstated. Those are exactly the properties a tool-bearing role cannot accept and a tool-free eval can.

Conditions on the download:
1. Fetch only `GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q8_0-MTP.gguf` (29,069,202,688 B) pinned at repo sha `57c00f892e6b2ac425eb34b3ab12b3025901fb66`. Do **not** fetch any `.ninfer` file (undocumented binary container, loadable only by a third-party engine), `mmproj`, or the separate `mtp-*.gguf` (the Q8_0 file already carries the Q8_0 MTP head at blk.64).
2. After download, compute sha256 and require `b24e8c5fe3f2b282cce841400c740fcaca48afcdcf9e991f7bbe9a59a78cabfe` (= HF LFS oid = repo `GGUF/SHA256SUMS`). Note this match is **self-consistency, not independence**: both values come from the uploader's repo. Then re-run `gguf_header_review.py` on the local file and require the header digests recorded in section 4.
3. Disk/download timing is the operator's call (29 GB; no external event is named beyond that).
4. `uploader_class = community`. Registry entry (if ever created) carries `tools: none`; evaluation uses tool-free suites only until SEC-EXEC-1..3 land (as AMSB-C390-1 already states).

## 1. Uploader identity and history (SEC-ART-3 field: provenance)
| Item | Finding |
|---|---|
| Account | HF user `nerkyor`, display name "Lynn"; created 2026-04-25; 26 models, 0 datasets, 0 spaces, 0 orgs; 400 followers, follows 0; 28 discussions; bio: "Post-training Qwen models for coding and efficient reasoning." |
| Verified link to a real-world identity | **None on HF** (no org, not Pro, no verified badge in the API). The card links an app "Lynn" (GitHub `MerkyorLynn/Lynn`, download site `merkyorlynn.com`) and a ModelScope user `Merkyor`; the GGUF header's `general.name` is `Merkyor/Qwen3.8-27B-Coder390-EfficientThink`. The `nerkyor` / `Merkyor` / `MerkyorLynn` spelling variation is almost certainly one person but the HF-to-GitHub link is **one-directional (card claims it; not independently verified)**. |
| History | Continuous 6-month record: Qwen3.5/3.6/3.8 imatrix GGUFs, "DSV4Pro Thinking Distill", "Heretic" uncensored variants (2026-06 to 2026-09), then this lineage. Earliest repos have 59-130 downloads; this repo 318,292 downloads / 457 likes (popularity, not evidence of safety). |
| Third-party scrutiny | 13 discussions, 12 open; no security or integrity report; one bug (DFlash2 + vLLM prefix caching, outside our stack); one naming critique (answered by uploader). |
| Card hygiene | The card is 282 KB, promotes a desktop app with `.exe`/`.dmg` download links. Irrelevant to GGUF loading, noted as a social-engineering surface: **never follow those links**. |
| Class | **community** (not first-party Qwen, not an allowlisted re-quantizer). |

## 2. Repo file list vs claimed base (SEC-ART-9-style relay/sidecar check)
Claimed base chain: official `Qwen/Qwen3.8-27B` -> `nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-...-SimPO-DFlash2` -> this repo (SFT + RLOO rounds). It is a **post-train, not a relay**, so byte-identity to upstream is not expected and is not claimed.
- Extension census of the full tree (220 files): .json 94, .safetensors 60, .ninfer 15, .gguf 11, .jinja 8, .png 6, .md 3, .yaml 2, extensionless 21 (SHA256SUMS files, .gitattributes). **Zero** `.py`, `.pt`, `.pkl`, `.bin`, `.sh`, `modeling*`/`tokenization*`, `adapter*`, `lora*` files. `config.json` has no `auto_map` (no `trust_remote_code` path). **No LoRA/adapter files.**
- Architecture: `Qwen3_5ForConditionalGeneration`; BF16 `config.json` differs from the official one only in `dtype`, `use_cache=false`, `transformers_version` 5.17.0 vs 5.8.0.dev0, and vision rope/model_type labels (no hidden-size/layer/vocab/MTP differences).
- Non-standard artefacts: 15 `.ninfer` packages (custom container for a third-party engine, `iamwavecut/ninfer-all` / `Neroued/ninfer`; 4 are "NOT-FOR-V3" V100 variants). Not loadable by llama.cpp; **out of scope, never fetch**. `VLLM_LOAD.json`, `recipe.yaml`, `hf_quant_config.json`, `manifest.json` are small JSON/YAML descriptors.
- `BF16/tokenizer.json` is 19,989,325 B (LFS oid `06b9509352d2af50...`) vs official 12,809,320 B (`0997f410c57a1f4e...`): a re-serialised tokenizer (merges inlined, official ships separate `vocab.json`/`merges.txt`). It is not used by llama.cpp; the **GGUF-embedded** tokenizer was compared directly (section 4) and is clean.
- Sha chain: GGUF `SHA256SUMS` Q8_0 = `b24e8c5f...cabfe` = LFS oid from the tree API (match). Same for `mmproj` and `mtp-*.gguf`.

## 3. Licence and data provenance
- Declared Apache-2.0 "following the base model" (official `Qwen/Qwen3.8-27B` is Apache-2.0, LICENSE file confirmed). Weights licence is therefore fine **for the weights**.
- **Not covered:** (a) training data: SFT/SimPO/K3-continuation corpora are unsized, unreleased, undescribed; only the RLOO set is sized (172 groups / 1,376 trajectories). (b) Teachers (Opus5.5, GPT6Astra, Grok4.7, DSV4Pro, K3) "wrote gold answers and teacher trajectories"; the card gives **no terms-of-use statement** for those outputs. (c) Predecessor tagged `uncensored`/`heretic`; the predecessor card documents "refusal-tendency dissolution" with a self-reported **4/140 (2.9%) refusal rate**, i.e. refusal behaviour was removed upstream of this model and the card never states whether the RLOO/SFT rounds restored any of it. (d) No decontamination statement. The AMSB licence waiver (L547) covers licence terms, **not backdoor provenance**; this record is the backdoor-provenance answer: *unauditable*, not *clean*.
- Consequence: acceptable for a sandboxed, tool-free eval whose output never reaches a tool or the serving stack; unacceptable for a tool-bearing role (SEC-ART-3 gate) until the data side can be audited.

## 4. GGUF header check (Q8_0-MTP, first 16 MiB, metadata only)
Reproduce: `gguf_header_review.py` (this directory). Compared against our local base `/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf` (header read only).
| Check | Result |
|---|---|
| Magic / version / counts | GGUF v3, 866 tensors, 39 KV |
| `general.architecture` | `qwen35` (matches card and our base) |
| `qwen35.block_count` | 65; `nextn_predict_layers` 1; `blk.64` holds `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` plus a full attention+FFN layer, identical names and shapes to our base's MTP block |
| Tensor set vs our base | **Names and dims identical for all 866 tensors.** Only type differs: 96 tensors (`blk.*.ssm_alpha.weight`, `blk.*.ssm_beta.weight`, 48 each) are BF16 here vs Q8_0 in our base (consistent with the card's "SSM control branches stay BF16"; "Q8_0" label is therefore not uniform Q8_0, size 29.07 GB) |
| Tokenizer arrays | `tokens` (248,320), `token_type`, `merges` (247,587): **sha256 identical to our base** (`6d8afc30...`, `d54439ef...`, `b645cd08...`). Tokenizer.ggml.model `gpt2`, pre `qwen35`. |
| Padding token id | 248044 here vs 248055 in our base GGUF; 248044 equals the official `generation_config.json` pad id, so the difference is on our base's side, not a defect here |
| Embedded chat template | 8,952 B, sha256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` = **byte-identical to the official `Qwen/Qwen3.8-27B` `chat_template.jinja`** and to this repo's own `BF16/chat_template.jinja`. No injected system text, tool-call rewriting, or hidden instructions (full text compared by sha256, official fetched as text). Our served base template (9,993 B, unsloth-patched: merged system messages, `high`->`xhigh` alias) differs from official; the A/B must pin which template each arm uses. |
| Sampling metadata | `general.sampling.top_k 20, top_p 0.95, temp 1.0` = official `generation_config.json` |
| Metadata strings | `general.name` `Merkyor/...` (spelling variant, see section 1); no URL/command strings in KV; no imatrix block |

## 5. Launch-flag label-vs-function check (llama.cpp GGUF section of the card, against v10 `ffc1bac82` sources)
| Recommended flag | Label | Actual function in v10 | Alters generation? |
|---|---|---|---|
| `--jinja` | use template | uses the embedded template (`common/arg.cpp:3438`); embedded = official | no (template verified) |
| `--spec-type draft-mtp` | MTP speculation | exists (`common/speculative.cpp:41`, `arg.cpp:3964`); speculative decoding, distribution-preserving by design | no (acceptance only) |
| `--spec-draft-n-max 4` | draft length | exists (`arg.cpp:3890`) | no |
| `-md <mtp gguf>` (optional) | external Q8 head | `--spec-draft-model/-md` (`arg.cpp:3956`) | no; not needed for Q8_0 |
| `-c 409600 -np 4` | context/slots | 102,400 tokens per slot, **below** the official 262,144 reasoning cap | no, but caps reasoning length: the A/B must set caps explicitly |
| `--temp 1.0 --top-p 0.95 --top-k 20` (16G command only) | sampling | matches official + embedded defaults | no |
| `--mmproj` | vision | not used (text-only A/B) | n/a |
No flag silences safety behaviour, edits the prompt, sets logit bias or grammar, or injects chat-template kwargs. The headline "reasoning effort xhigh" is the template default (official default is also `xhigh`). Risky item: the card's benchmarks used SGLang + DFlash2 and a 94,208-token cap; none of its numbers transfer to our harness (already recorded in intake-2193).

## 6. Residual risks not closable without compute or weights
- Behavioural backdoor / trigger behaviour in weights: **not assessable** from metadata. The Microsoft AIRT scanner (SEC-ART trigger line) has no public implementation yet; keep the probe for tool-bearing review.
- File integrity independent of the uploader: only possible post-download (hash vs LFS oid), which proves transport integrity only.
- MTP head is the official head unchanged on a post-trained backbone (acceptance not guaranteed); log MTP/DFlash2 acceptance against base as required by the trigger line.

## 7. Draft handoff text
`AMSB-C390-0`: tick (review complete, 2026-10-21): GO conditional for Q8_0 download as tool-free eval artifact only; record `artifacts/intake-backdoor-cvec-20261021/AMSB-C390-0-review-record.md`; hash verification of the downloaded file remains a step of the download itself.
