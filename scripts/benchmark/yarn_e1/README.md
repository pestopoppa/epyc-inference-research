# E1 historical runners and proof conformance

These reviewed-copy candidates preserve the original E1 runners without overwriting workspace-ec scratch or the separate NI YARN-E1-MEM packet. PROVENANCE.json pins the original bytes; tests/fixtures/A1.server.log is the exact retained log. The shared helper tolerates timestamp interruptions in metadata assignments, restricts log proofs to the last launch, and retains forbidden-line refusal. Both CPU and GPU paths import it; both launch paths share proof visibility flags.

Run offline: `python3 scripts/benchmark/yarn_e1/tests/test_yarn_proof.py`. No models, network, processes, GPU or production source are used. The actual preflight is tested against the retained log with only unrelated live inspection stubbed; legacy matching fails two required proofs.

Runtime deployment is separate: provision original frozen inputs/vendor assets beside the runner before real work; launch_arm.sh defaults YARN_E1_HOME to this directory and permits an explicit provisioned home. No launcher was run or installed by this proposal. Existing original hardware/model paths and window guard are historical recipes; requalify before any run. This source acceptance does not assert fresh E1 results or modify NI MEM-prep.
