handoffs: Nemotron 3.5 Bolt pending fold for the champion owner (2026-10-21)

experimental/champion-plus-nemotron-bolt-20261021 @ 59c1b915f on dd3d9721c
(fork): per-layer MoE hparams (port of upstream #25444/#28323/#30520),
saver key-clobber pick (#26693), mtmd nemotron_omni projector. Builds
{gpu,cpu}-20261021-59c1b915f-nemotron; greedy identity 3/3, op tests and
test-llama-archs clean; MI210 pp512 899 / tg128 53.4 t/s (indicative).
DFlash LiLiCorr drafter not supported (rides CH-18). Belief-kernel source
row + VB-NEMOTRON-BOLT-FOLD-WIRE task; progress note.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
