---
name: security-review
description: Use when reviewing epyc-inference-research diffs, commits, dependencies, benchmark runners, dataset/download scripts, generated artifacts, logs/results, shell scripts, or frontmatter/YAML for exploitable security risk. Emits only exploit-path-gated findings with P0-P3 severity.
---

# Security Review

Use this skill for focused security review of `epyc-inference-research`. Report only findings with a plausible exploit path, or say no exploitable issue was found.

## Scope

Review these areas first:

- `scripts/benchmark/` benchmark runners, scorers, adapters, manifests, and helper scripts.
- `scripts/security/` audit and repository-safety logic.
- Dataset and download flows, especially `scripts/benchmark/download_*.py`, `scripts/corpus/`, and any script that fetches, extracts, caches, or rewrites data.
- Generated artifacts and local outputs under `benchmarks/results/`, `benchmarks/evidence/`, `data/`, `logs/`, `research/*results*`, and any checked-in run logs or summaries.
- Shell scripts under `scripts/**/*.sh`.
- Dependency and environment changes in `pyproject.toml`, `uv.lock`, `.github/dependabot.yml`, and any repo-local setup or config file.
- YAML, manifests, and frontmatter-bearing docs that control execution, data selection, or result publication.

If the diff is narrow, inspect only the files and call paths needed to trace an exploit path across these surfaces.

## Review Model

Treat the repository as a research environment with mixed trust:

- Benchmark inputs may be attacker-controlled if they come from datasets, prompts, model outputs, logs, or checked-in artifacts.
- Local logs and generated results can be re-consumed by scripts, notebooks, or agents and should be treated as potentially tainted.
- Download scripts may cross trust boundaries from the network into the filesystem.
- Dependency changes may execute code during install, build, or import.
- Shell scripts are privileged sinks when they pass filenames, model names, URLs, or config values into subprocesses.

## Workflow

1. **Scope and evidence**
   - Identify entrypoints, changed files, new dependencies, config changes, shell wrappers, generated artifacts, and data files.
   - Use exact string search and only read the code needed to trace the flow.
   - If the review scope includes frontmatter, YAML, or manifests, inspect those first because they often steer execution or data selection.

2. **Pass 1: Candidate discovery**
   - Trace data flows across trust boundaries: dataset -> parser -> scorer, download -> extract -> filesystem, config -> shell/subprocess, log/result -> downstream consumer, dependency -> install/build/runtime.
   - Apply STRIDE: spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege.
   - Apply supply-chain checks for dependency changes, generated code, install scripts, broad version ranges, vendored binaries, and lockfile drift.
   - Watch for exploit primitives that are especially relevant in this repo:
     - `shell=True`, `eval`, `exec`, `os.system`, or unquoted shell expansion.
     - Unsafe archive extraction, path traversal, symlink following, or overwrite of checked-in files.
     - Network fetches that trust redirects, schemes, hostnames, or content types too broadly.
     - Result/log parsing that can be poisoned by attacker-controlled model output or artifacts.
     - Prompt or tool injection when logs, datasets, or generated summaries are re-fed into agents.
     - Secrets or credentials written into checked-in outputs, manifests, or audit logs.

3. **Pass 2: Exploit validation**
   Promote a candidate only if all gates pass:
   - A realistic attacker or compromised input source can reach the path.
   - The path crosses a trust boundary or weakens a security invariant.
   - A vulnerable sink or privileged action is reachable.
   - Existing validation, sandboxing, allowlists, or deployment constraints do not already block it.
   - The impact is concrete: data exposure, unauthorized action, code execution, durable agent/tool compromise, integrity loss, availability loss, or secret leakage.
   - A minimal fix is clear.
   - File and line evidence exists.

   If any gate is missing, do not promote the candidate. Mention it only under residual risk if it is worth tracking.

4. **Validation before reporting**
   - Run `git diff --check`.
   - Validate YAML and frontmatter in any touched `.md`, `.yml`, or `.yaml` files.
   - Run the lightest syntax checks available for touched code:
     - `python -m py_compile` or `python -m compileall` for touched Python files.
     - `bash -n` for touched shell scripts.
   - Prefer targeted checks over broad test suites unless the change truly spans multiple execution paths.

## Severity

- `P0 / Critical`: low-friction RCE, credential or key exfiltration, broad data exposure, privilege bypass, or durable controller/tool compromise.
- `P1 / High`: authenticated privilege escalation, scoped secret disclosure, SSRF to sensitive systems, tool/subprocess injection, or likely malicious dependency execution.
- `P2 / Medium`: narrower security invariant break, realistic DoS, unsafe agent/tool behavior behind specific conditions, or missing validation on a sensitive path.
- `P3 / Low`: defense-in-depth gap with a credible but limited path, logging or audit weakness, or a hardening issue without immediate sensitive impact.

Do not assign P0-P2 without a concrete exploit path.

## Output

Lead with findings ordered by severity:

```markdown
- [P1] Imperative title under 80 chars
  - Location: path/to/file.py:123
  - Problem: Security invariant that is broken.
  - Exploit path: Attacker input -> trust boundary -> sink -> impact.
  - Suggested fix: Minimal safe change.
```

Then include:

- **Residual risk**: candidates that did not pass exploit gates, uncertainty, or follow-up checks.
- **Checks run**: commands, files, and code paths inspected.

If no findings pass the gates, state that explicitly and name the highest-risk surfaces inspected.

## Guardrails

- Do not expose secrets found during review; identify the path and remediation class only.
- Do not run exploit payloads against live services unless explicitly asked and isolated.
- Do not mutate generated artifacts, indices, or runtime state while reviewing.
- Prefer narrow fixes: validation, capability checks, path guards, permission checks, dependency pinning, sandboxing, output encoding, or fail-closed gates.
