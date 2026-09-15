# AutoKernel headline admissibility (S3-AKU-03)

An AutoKernel champion-versus-production headline requires **28 independent process
launches: 14 anchor and 14 candidate**. A smaller comparison is retained as measurement
evidence but headline publication refuses; internal repetitions within one launch do not
increase `n`.

## Why 28

The preregistered minimum detectable effect is **3.0%**. This is the campaign's existing
operational screening scale: effects smaller than 3% may still be useful evidence, but
this contract does not spend enough independent launches to advertise them as absolute
headlines. Making that choice explicit prevents the required count from silently tracking
a convenient caller default.

R23-57 / INF-70 RETEST-1 measured between-process-launch SD as **2.793%**. We use the
standard equal-size, two-independent-sample normal planning approximation, two-sided
`alpha=0.05`, power `0.80`:

```text
n_per_arm = ceil(2 * ((z_0.975 + z_0.80) * sd / MDE)^2)
          = ceil(2 * ((1.959964 + 0.841621) * 2.793 / 3.0)^2)
          = 14
```

Thus `N=28` total. The emitted interval is a deterministic two-sided 95% paired-bootstrap
percentile interval over 20,000 whole A/B launch blocks. It targets the headline's exact
median(candidate)/median(anchor)-1 estimator while preserving the alternating design's drift
control (`paired_bootstrap_median_ratio_95_seed_20260915_draws_20000`).
The floor and headline records carry `unit`, `n`, `n_per_arm`, MDE, measured SD, alpha,
power, sidedness, planning method, CI level, and CI method. This is a planning convention,
not a retrospective claim that launch rates are exactly normal.
