# Dropping in a hypothesis

You have an idea about what would make the kernel faster. Write it in a file. The
agents pick it up from there, and the record tells you what they did with it.

**You do not have to write a falsifier.** An agent has to write one before it can spend
a resource claim on your idea — that is their job, not yours.

---

## Add one

The file is a JSON object with two keys. This is a complete, valid file:

```json
{
  "schema": "epyc.autokernel.operator_hypotheses.v1",
  "hypotheses": [
    {
      "hypothesis_id": "akh-fuse-norm-cluster",
      "statement": "the elementwise/norm cluster is where the B=128 decode time goes; fusing it should be worth 15%"
    }
  ]
}
```

That is the minimum: **an id and a sentence.** The id must start with `akh-`.

JSON, not YAML, deliberately: YAML would read a falsifier of `no` as the boolean
`False` and `1.5-3` as something that no longer says what you wrote.

### The four other fields you may use

| field | what it does |
|---|---|
| `falsifier` | one line that would show you were wrong. Write one if you have one; leave it out if you do not. Do **not** write `"tbd"` — that is refused, because it is an empty string wearing a hat. |
| `regime` | where the claim applies, e.g. `{"backend": "llama_gpu", "phase": "decode", "batch_band": "b128"}`. Worth writing: it is what stops a negative measured on CPU prefill from being used against a GPU decode idea. |
| `author` | defaults to `operator`. |
| `created_at` | free-form; nothing branches on it. |

### What the file will not accept

`priority`, `rank`, `weight`, `status`, `resolved`, `outcome`, `resolution`,
`evidence_grade`, `grade`, `origin`, `notes` — all refused at load, loudly.

The first three because this is not a queue-jumping mechanism: your hypothesis is a
proposal source, and it goes through the same critic and the same rejection conditions
as one the agents thought of. The rest because they are the *record's* fields, and a
file you can hand-edit must not be able to say a question was confirmed.

Every hypothesis enters at evidence grade `design_prior` — "worth considering" —
whoever wrote it. Being your idea is not evidence. It gets *tried*, not believed.

---

## What happens when an agent picks it up

1. **Tracked.** The next intake opens your entry in the ledger. If it has no
   falsifier, it is listed as *awaiting one* and no compute can be spent on it yet.
2. **A falsifier is written for it.** An agent proposes the predicate, records who
   wrote it and why. Your sentence is never rewritten — the proposal sits beside it.
3. **Adopted, and it leaves your file.** When an agent takes the idea on, the entry is
   removed from your file and ownership transfers to them. Your file comes back
   byte-for-byte as you typed it, minus that one entry: same indentation, same key
   order, same line breaks. Nothing else is touched.
4. **Attempted.** Every proposal dispositioned against it is recorded as an attempt.
   An attempt never closes the question — including when the proposal failed for some
   unrelated reason, which is the whole point.
5. **Resolved** `confirmed` / `refuted` / `inconclusive`, and only with the evidence
   that resolved it and an observation stated against the falsifier.

After that, the idea is *memory*: the same idea proposed again, reworded, is rejected
with a receipt. When the production kernel moves, that rejection lapses — a thing that
lost on v7 may win on v8, and the ledger is memory rather than a blacklist.

**Deleting your line does not close the question.** Once it has been tracked, the
record owns it; removing the text just means the file no longer mentions it.

---

## Find out what happened to it

```python
tracker.trace("akh-fuse-norm-cluster").answer          # from the record alone
tracker.trace("akh-fuse-norm-cluster", store).answer   # …and whether it is still in your file
```

One sentence, e.g.:

> akh-fuse-norm-cluster was opened at 2026-08-04T09:12:03Z (origin operator, author
> operator, entry grade design_prior); adopted by mainA at 2026-08-04T09:40:11Z (taking
> this into the fusion campaign), so it left the operator store at
> /…/operator_hypotheses.json and is now owned by the agents; falsifier (proposed by
> mainA): a current wall-share map shows the cluster under 20%; 1 attempt(s), 1 claim
> authorization(s); refuted at 2026-08-04T11:02:44Z by mainA on evidence
> ['ake-20260803-0001'] (protocol_bound): the wall-share map put the cluster at 11%,
> under the 20% line.

`trace(...).to_dict()` is the same thing with every field separately, including the
adoption record (which carries your original entry text inline, so the trace stands
alone even if the file is never read again) and every claim spent on it.

To see everything at once: `tracker.still_open()`, `tracker.resolved()`, and
`do_not_repeat.planner_round_block(tracker, ledger, round_id=...)`, whose
`awaiting_falsifier` list is the work the loop owes you.

---

## Two things this does not do yet

* **Nothing runs it on a schedule.** `campaign.py` — the entrypoint that actually
  starts a run — does not reach this module, so intake, adoption and the round block
  happen when an agent working the campaign calls them. Your file is read; it is not
  polled.
* **An idea with no `mechanism` in its regime cannot be checked against memory.** The
  do-not-repeat ledger matches on *what change is being made*, and a one-line idea
  usually does not name one. That case comes back COULD_NOT_CHECK, not "clear" — the
  claim is still allowed (a wrongly suppressed idea is silent and permanent; a wasted
  re-run is loud and costs one claim), and the uncertainty is recorded on the spend.

---

Mechanism, and every refusal above with the reason it exists:
`controller/hypotheses.py`. The memory plane: `controller/do_not_repeat.py`. The whole
path, tested end to end over real files:
`controller/test_hypothesis_path_end_to_end.py`.
