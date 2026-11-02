# FA-4b response integrity replay

The independent successor report revalidated the retained M1/M2 manifests,
record seals, request/response and server-log links, completed IDs and native
scorer hashes, then counted JSON numeric finiteness and the existing DF2-9
text-garbage rule. It found zero nonfinite JSON responses and zero matching
garbage strings across 2,400 accepted responses (200 scored and 100 greedy in
each arm and configuration). Failed or missing records cause refusal rather
than silently becoming zero counts. This does not establish that model logits
or hidden state were finite, and the previous 1-of-4 quality-equivalence
result is unchanged.

The complete report is `/mnt/raid0/llm/tmp/rtg57-fa4b-accounting-20261102/REPORT-check.json`,
SHA-256 `fbe1f22ea5db84b5bdcb5cabd4bb5ee68b352882272c17101b134667db4c72a2`.
The existing report is identical after removing the new count and scope fields.
The M1/M2 scripts are byte-identical to research `b607e1901db92952dc1b4946529d6624018cabb3`;
the scorer snapshot is orchestrator `e1e3e59d72f9be800d7c3ba4cb1cf10fdd4a5c1c`,
verified against the original sealed source hashes. The report is an offline
replay of retained observations, not a new inference run or a new quality gate.
