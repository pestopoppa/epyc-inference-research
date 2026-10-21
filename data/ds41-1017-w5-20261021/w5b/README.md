# DS41-W5b: empty-EOS on article + "Question: ...?\nAnswer:" raw completions (2026-10-21)

## Symptom (W5 rows `../raw/coh-*.rows.jsonl`)

- The Sept Q4 and 1017 arms both emit token 1 (`<｜end▁of▁sentence｜>`) as the FIRST generated token on
  the article prompts at 309 / 600 / 1149 / 2041 prompt tokens.
- In the same runs, `q40` passes at 41 tokens (`Question: ... time.\nAnswer:`), and so does `p220` at
  235 tokens (`... final list.\n\nAnswer:`).
- The question-first prompt `qfirst9000`, which ends `The subject of the article is`, passes at 2030
  tokens.
- Two different finetunes on two different expert kernels (Sept: iqk Q4_K; 1017: the MXFP4 repack)
  fail identically. That argues against an expert-kernel defect.

## Static findings (no compute)

- **Tokenizer**: llama.cpp ids are identical to HF `tokenizer.json` (1017 @ `1bae0e10`) on all 10 W5
  prompts after the BOS (`tok_parity.py`, vocab-only `llama-tokenize`, giving `tok_parity.json`). The
  pre-tokenizer regex in the GGUF (`pre=joyai-llm`, which maps to the deepseek3 regex) is the HF one.
- **BOS/EOS**: one BOS (id 0) per prompt, so no double BOS. `add_bos=true` in the GGUF matches
  `encoding.py`, which emits BOS explicitly. HF `tokenizer_config.add_bos_token=false` only concerns
  the HF tokenizer call. EOS is id 1, the same in the GGUF and HF.
- **Chat template**: not involved. `/completion` with a string prompt applies no template, and neither
  GGUF carries `tokenizer.chat_template`. The coherence harness (`../harness/coherence.py`) sends
  greedy `/completion` with no stop strings, `n_predict` 80 and `cache_prompt` false, so it has no
  harness defect.
- **The one token-level difference** between the failing and passing families: all failing prompts
  end with the merged token `?Ċ` followed by `Answer` and `:`. The passing ones end `.Ċ` or `.ĊĊ`
  followed by `Answer` and `:`. The failing family is also wikitext-formatted (` , `, ` @-@ `) and cut
  mid-word.
- What it is NOT: tokenization, double BOS, the chat template, or the harness. Model behaviour on this
  raw prompt shape, and a port long-range-attention effect above roughly 256 tokens, are both still
  open. The `qfirst` pass does not clear the port, because its answer is reachable from the last ~128
  tokens (the sliding window).

## Decisive probe: one target-only server, step W5B of the slot driver

`probes.py` runs 20 greedy requests with `n_probs=10` on one 1017 server at `-t 48` on 0-47, with no
drafter (DSpark serial leaves `n_probs` empty). It records P(EOS) of the first token and the top-10
candidates:

- **control**: the `len1200` prompt as run in W5.
- **suffix variants**: question ends `.`; `?\n\nAnswer:`; a trailing space; a forced ` The` prefix.
- **content variants**: no instruction line; a cut at a sentence boundary; de-wikitexted; question both
  first and last.
- **short controls**: `Question: What is the capital of France?\nAnswer:` with `\n` and with `\n\n`.
- **length ladder**: 450 / 700 / 950 / 1150 article characters with the same suffix.
- **EOS banned** (`logit_bias`) on `len1200` and `len9000`: does the model know the answer?
- **chat-rendered**: the same content through the epyc-v1 template (1017 `encoding.py`), non-thinking.

### How to read the result

| Outcome | Verdict |
|---|---|
| The short `?\nAnswer:` controls answer; the article family EOSes with high P(EOS) at every length; the chat-rendered prompt answers | Model behaviour on raw article+QA completions, not a port bug. Use chat-rendered or question-first prompts in long-prompt gates. |
| A sharp length threshold between ladder steps, near a compression-group boundary | Port long-range-path suspect. File it against DS41-T3 (reference parity) before any long-prompt gate. |
| The suffix variants (`.`, `\n\n`) fix it at the same length | A `?Ċ`+`Answer:` token-context effect, which is still model behaviour. |
