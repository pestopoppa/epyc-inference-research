# Security Review

Audit the specified scope in `epyc-inference-research` for exploitable security issues using the `security-review` skill.

**Target scope:** $ARGUMENTS

## Your Task

1. If `$ARGUMENTS` is empty, inspect the current diff first.
2. Review benchmark runners, dataset/download scripts, generated artifacts, local logs/results, shell scripts, dependency changes, and any touched frontmatter/YAML.
3. Report only findings that satisfy the exploit-path gates.
4. If no finding passes the gates, say so explicitly and include residual risk plus checks run.
