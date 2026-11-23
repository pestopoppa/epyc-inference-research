# Community intake review bundle

Prepare one JSON object with `claims`, a list of manually verified source claims.
Each claim has single-line `title`, `mechanism_id`, `target_symbol`, `statement`,
`falsifier`, `target_surface`, `url`, and `tier` (`upstream`, `community`, or `paper`). Supply
`achieved_gb_s` and `ceiling_gb_s` together only when the source reports a
comparable achieved rate and ceiling on its own hardware, and include a
`bandwidth_basis` describing that hardware and denominator. Missing values stay
unknown; rates above the ceiling are retained as a reason to scrutinize cache or
the source's denominator. Upstream candidates are ports to test on gfx90a.

For each sweep inspect upstream llama.cpp kernel commits/PRs since `0db32c06`,
community forks and writeups, and the existing research-intake paper records.
Provide the current target's hotspot export as a JSON list of `Hotspot.to_dict()`
rows, and the corresponding live `experiments.db`. Preserve the source URLs and
original measurement context in the claims. This tool does not fetch or verify
sources and cannot establish the freshness of the supplied hotspot export: the
reviewer must check both before using the bundle.

```
python3 -m autokernel.loop.community_intake --claims claims.json \
  --hotspots hotspots.json --experiments-db /path/to/experiments.db \
  --out review-bundle.md
```

Output is exclusive-create and outside the experiment store and any directory named `inbox`. Null lookup
matches exact mechanism IDs or target symbols and reports original epoch and target: absence of a
match is not proof of novelty. The operator reviews the bundle and manually
selects and injects any proposed Markdown seed sections. This command launches no experiment and changes
neither the store nor the inbox.

Optionally pass `--upstream-repo /path/to/llama.cpp --upstream-ref <verified-upstream-ref>`
to enumerate CUDA/HIP-path commits since the freeze for triage. The operator
must ensure the local ref is current and inspect each commit and PR thread; path
changes are candidates, not a claim that every change is an improvement.
