# SPEC-COPY-COV-1 predeclaration (sealed before any run, 2026-11-24, autokernel)

Box: epyc-root `handoffs/active/speculative-decoding.md` SPEC-COPY-COV-1 (intake-2847#00, #01, intake-2832#00, #01).
Class: non-inference, offline. Gate class: futility-only. No timing claim.

- **Question.** Would an exact longest-suffix copy index find >= 5-token matches whose next 3 tokens are right
  often enough on archived agentic traffic to justify a copy-proposal drafter probe?
- **Input.** `epyc-root/artifacts/copyspec-20261006/source-archive/results/P.jsonl` (85 requests: 40 debugbench =
  code, 45 hotpotqa+general = prose) and the matching prompts in `prompts_p1.jsonl`. Only `ok` rows with text count.
- **Tokens.** Qwen3.6-35B-A3B-MTP-Q8_0 vocab (the model that produced P), via the C21 CPU store `llama-tokenize`
  (vocab-only load), `--no-bos`. The prompt is the raw user text, without the chat template; template tokens do not
  carry copyable content.
- **Measure.** For each output position: the longest causal suffix with an earlier occurrence (suffix automaton), the
  3 tokens after its first earlier occurrence, and how many leading ones agree with the actual output.
  Variants: self, loo (decisive), insample (leaky upper bound).
- **Sample size.** All 85 requests, every position (~10^4 positions). The share is a census of this archive and is
  not a sample estimate; the shortfall is coverage of other traffic, and that is stated, not bootstrapped.
- **Rule.** hit = len >= 5 AND 3 agreed. On loo: code share < 5% AND prose share < 2% -> CLOSE_COPY_PATH;
  otherwise FIRE_TRIGGER (the box's trigger: a default-off copy-proposal probe on DFlash2, one GPU row, and a
  CoPyro arm in DF2-FT-1). An empty class -> NOT_JUDGEABLE_EMPTY_CLASS.
- **Stop.** One pass answers it. There is no interim look: the replay is deterministic and runs in under a minute.
