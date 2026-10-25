#!/bin/bash
# Read-only descendants of a captured job PID; never select by global name search.
lb_pid() {
    local root=${1:?captured root pid required} p comm children
    [[ $root =~ ^[0-9]+$ ]] || return 1
    IFS= read -r comm < "/proc/$root/comm" 2>/dev/null || return 1
    if [[ $comm == llama-bench ]]; then printf '%s\n' "$root"; return 0; fi
    # A process may disappear between reads. That is no sample, never a wrapper.
    for children in /proc/"$root"/task/*/children; do
        [[ -r $children ]] || continue
        for p in $(cat "$children" 2>/dev/null); do
            lb_pid "$p" && return 0
        done
    done
    return 1
}
