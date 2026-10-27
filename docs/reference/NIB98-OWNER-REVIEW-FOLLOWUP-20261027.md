# NIB98 owner-review follow-up — 2026-10-27

## Scope

This note records the tests-only owner-review follow-up to the DS41 lifetime admission proposal. It does not establish production adoption, universal loader coverage, or actual `llama-server` descriptor retention.

## Source and test evidence

- Proposal: `codex/nib98-atomic-admission` at `036b8a51957cb6366afb490db34a468e1d482055`, based on `dc77ce388`.
- Follow-up: `codex/nib98-owner-review-20261027` at `4c4ad160c783dcda452c63c6bc18426f21af6f89`, based on the proposal. The commit changes only `scripts/kernel_rnd/autokernel/loop/test_ram_admit.py` (39 insertions, 7 deletions). The test file SHA-256 is `2305b901df63725c7d2a37257e466e6c75c43c050444d6b16b4fde656804fedd`.
- The focused test run passed 20 total tests. The follow-up replaces string-presence coverage with admission-order and live-FD checks, including three parameterized cases and strengthened launch-site call checks. The bounded run record is `/mnt/raid0/llm/tmp/refactor-01a16fc3/nib98-review-fixes-20261027/OWNER-PACKET.md` (SHA-256 `5fc3ea1deccff430fa2cf591721c00a49432cea2ae0ccb65fac89fe379cdfb5d`).
- Four proposed shell patches were tested under strict shell behavior for refusal/error controls and passed; all four remain unapplied. These are control results for proposed patches, not evidence that the patches are deployed.
- Owner reply confirms exclusive serialization: one DS41 server at a time; a simultaneous second server is refused by the lease design. The reply is in `/mnt/raid0/llm/tmp/refactor-01a16fc3/compute-coordination-20261026/owner-reply.md` (2026-10-27 16:00Z section).

## Adoption boundary

No common inode was provisioned and no live guard or launcher changed. The test evidence does not show that an actual `llama-server` retains the descriptor through its lifetime. Before activation, the owner must complete the loader/adopter inventory and an approved canary proving actual server FD retention. Existing loaders outside the reviewed call sites remain outside the proposal's guarantee. No model run or runtime safety claim follows from this follow-up.

## Conclusion

The owner-review follow-up strengthens test coverage and confirms the intended exclusive one-server policy. It is a tests-only proposal update. The proposal remains unadopted pending the specified inventory and canary.
