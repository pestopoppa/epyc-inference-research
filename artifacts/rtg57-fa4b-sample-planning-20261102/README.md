# FA-4b prospective sample-size sensitivity

This is a planning simulation, not a revised protocol, measured result, or
power guarantee. It fixes each cell's future ON-only/OFF-only probabilities at
the empirical values from the original 200 paired scores, simulates independent
new questions, then uses the existing 10,000-draw paired percentile-bootstrap
indices and ±2 pp acceptance margin. The output's Wilson intervals describe
only the finite simulation count, not uncertainty in the very sparse observed
discordance rates or future task distribution.

With 500 seeded simulations per cell and size, the proportion of simulated
intervals inside the original margin at 200 / 400 / 800 / 1600 pairs is
q8/off `.378/.756/.922/.998`, q8/on `.412/.588/.722/.946`, f16/off
`.162/.262/.274/.378`, f16/on `.274/.712/.934/1.000`. In this point model,
f16/off's observed +1.5 pp delta is close to the +2 pp margin and requires more
than simply assuming 400–800 pairs will resolve it. These numbers are sensitive
to the tiny original discordance counts and must not be mistaken for actual
study power. No threshold or corpus is ratified by this calculation.

The historical score outputs hit the 512-token cap in 102–105 of 200 responses
per arm/config. The report contains no counterfactual uncapped pairs, so the
quality change, runtime and sample size required under an uncapped regime
cannot be inferred from it. A future design should separately declare its cap,
corpus, sampling rule and cost, and keep exposed original records intact.

Input M2 `REPORT.json` SHA-256
`edba0cccca5658887bb6de62448fdf3e28ccab0b747de8eced9862eedc1371cb`;
`planning.json` SHA-256
`359d447660921d4542f99937e8fb00590c123dc2f0586bcd86db5ce749504c00`;
generator SHA-256 `b5b45e93aefb28cc42f48d169679f48bb2df196b3377e7b1370cbcff820d0d96`.
No model or GPU was used.
