#!/usr/bin/env bash
# placement_sampler.sh - in-window NUMA placement sampler for bench_canonical.sh (sourced, not executed).
#
# Extracted from bench_canonical.sh so it can be tested model-free (fake process + fake checker,
# test_placement_sampler.py). Behaviour history:
#   INF-70/C7  in-window proof; DS41-T6c  descendant walk fix (see the note above largest_rss_descendant).
#   PF1-EXP04-PLACEMENT (2026-11-03, r4 evidence): the sampler required TWO consecutive strictly
#   non-growing RSS samples at a 3 s cadence. On the pinned-host-weight GPU arm (A2, -ngl 99 -ncmoe 48,
#   load ~90 GiB then a ~10 s prefill) RSS kept creeping up through compute-buffer setup until the process
#   exited, so NO in-window sample was captured on either L (largest RSS seen 83 GiB, 1 stable sample).
#   Fixes: (1) 1 s cadence; (2) growth within PLACEMENT_STABLE_TOL_PCT (0.5%) counts as stable;
#   (3) PROVISIONAL samples every PLACEMENT_PROVISIONAL_EVERY ticks once the floor is reached, promoted
#   to the proof if the run ends before the RSS settles - still a sample taken DURING the run, labelled
#   "provisional" in the reason file; (4) COVERAGE: a sample whose numastat Total is < PLACEMENT_MIN_COVERAGE_PCT
#   (90%) of the process VmRSS is rejected (r4 A0@2048 "PASSED" on 19.4 GB of an 88 GB resident set), kept as
#   <log>.attempt-N, and sampling continues. No threshold on the node share lives here: that cap
#   (CANONICAL_MAX_NODE_SHARE_PCT) is passed in as $MAX_NODE_SHARE and is human-amendment-only.
#
# Inputs (globals): BINARY MODEL PLACEMENT_CHECK MAX_NODE_SHARE PLACEMENT_LOG PLACEMENT_REASON
# Env overrides (tests / tuning): PLACEMENT_SAMPLE_INTERVAL (s, default 1), PLACEMENT_FLOOR_PAGES
#   (default 1 GiB of 4 KiB pages), PLACEMENT_STABLE_TOL_PCT (0.5), PLACEMENT_PROVISIONAL_EVERY (10),
#   PLACEMENT_MIN_COVERAGE_PCT (90), PLACEMENT_MAX_ATTEMPTS (4).
# Outputs: PLACEMENT_LOG + PLACEMENT_LOG.rc on success; PLACEMENT_REASON always.

# Find the largest-RSS process in OUR OWN descendant tree. Never search by name
# across the host (shared box; a name pattern is a wildcard over other sessions'
# processes) - walk /proc/<pid>/task/<pid>/children instead. The model loader is
# always the biggest thing we started.
#
# DS41-T6c ROOT CAUSE (2026-09-23): the previous version enqueued children with
#     read -r kids < "$f" || kids=""
# and /proc/<pid>/task/<tid>/children is emitted WITHOUT a trailing newline, so
# `read` returns 1 at EOF *after having already assigned the pid list* - and the
# `|| kids=""` guard then threw that list away. Reset FIRST, read SECOND, and
# swallow the EOF status without touching $kids.
largest_rss_descendant() {
    local root="$1" best="$1" bestrss=0 rss kids k f
    local -a queue=("$root")
    local i=0
    while [[ $i -lt ${#queue[@]} ]]; do
        local cur="${queue[$i]}"; i=$((i + 1))
        [[ -r "/proc/$cur/statm" ]] || continue
        rss=$(awk '{print $2}' "/proc/$cur/statm" 2>/dev/null) || rss=0
        [[ -n "$rss" ]] || rss=0
        if [[ "$rss" -gt "$bestrss" ]]; then bestrss="$rss"; best="$cur"; fi
        for f in /proc/"$cur"/task/*/children; do
            [[ -r "$f" ]] || continue
            kids=""
            read -r kids < "$f" || true
            for k in $kids; do queue+=("$k"); done
        done
    done
    echo "$best $bestrss"
}

# Diagnostic dump of our own descendant tree, for the loud-failure path only.
descendant_tree_dump() {
    local root="$1" cur kids k f
    local -a queue=("$root")
    local i=0
    while [[ $i -lt ${#queue[@]} ]]; do
        cur="${queue[$i]}"; i=$((i + 1))
        [[ -r "/proc/$cur/statm" ]] || continue
        printf '  pid=%-8s comm=%-16s rss_pages=%s\n' \
            "$cur" "$(cat "/proc/$cur/comm" 2>/dev/null || echo '?')" \
            "$(awk '{print $2}' "/proc/$cur/statm" 2>/dev/null || echo '?')"
        for f in /proc/"$cur"/task/*/children; do
            [[ -r "$f" ]] || continue
            kids=""
            read -r kids < "$f" || true
            for k in $kids; do queue+=("$k"); done
        done
    done
}

# placement_coverage_pct FILE -> integer percent of the process VmRSS that the sample's numastat "Total" row
# covers; -1 when either number is missing (a synthetic/offline sample has no VmRSS: caller treats -1 as "cannot
# judge" and accepts only if the checker exited 0 or 3).
placement_coverage_pct() {
    awk '
        $1 == "Total" && NF >= 3 { tot = $NF + 0 }
        /^VmRSS:/ { rss_kb = $2 + 0 }
        END {
            if (tot > 0 && rss_kb > 0) printf "%d\n", (tot * 1024.0) / rss_kb * 100.0
            else print -1
        }' "$1" 2>/dev/null
}

# Sample placement while the process is alive. A sample taken after the run is
# not evidence: the pages are gone.
#
# Selection stays structural (our own descendants only). The expected `comm` is
# used as a CHECK on the pid the walk already chose - never as a search key.
sample_placement_in_window() {
    local root="$1" prev=0 stable=0 cand rss best_seen=0 cand_comm expect_comm
    local floor_pages="${PLACEMENT_FLOOR_PAGES:-$((1024 * 1024 * 1024 / 4096))}"   # 1 GiB
    local interval="${PLACEMENT_SAMPLE_INTERVAL:-1}"
    local tol_pct="${PLACEMENT_STABLE_TOL_PCT:-0.5}"
    local prov_every="${PLACEMENT_PROVISIONAL_EVERY:-10}"
    local min_cov="${PLACEMENT_MIN_COVERAGE_PCT:-90}"
    local max_attempts="${PLACEMENT_MAX_ATTEMPTS:-4}"
    local ticks_above=0 attempts=0 prov_rc="" prov_pid="" prov_comm="" prov_rss=0 prov_cov=-1 last_cov=-1 rc cov grown
    local attempt_log
    expect_comm="$(basename "$BINARY")"
    expect_comm="${expect_comm:0:15}"   # /proc/<pid>/comm is 15 chars max
    rm -f "${PLACEMENT_LOG}.provisional" "${PLACEMENT_LOG}.provisional.rc"
    while kill -0 "$root" 2>/dev/null; do
        read -r cand rss < <(largest_rss_descendant "$root")
        [[ "$rss" -gt "$best_seen" ]] && best_seen="$rss"
        if [[ "$rss" -ge "$floor_pages" ]]; then
            ticks_above=$((ticks_above + 1))
            # stable = not grown by more than tol_pct since the previous tick
            grown=$(awk -v r="$rss" -v p="$prev" -v t="$tol_pct" 'BEGIN{ print (p > 0 && r > p * (1 + t / 100.0)) ? 1 : 0 }')
            if [[ "$prev" -gt 0 && "$grown" -eq 0 ]]; then stable=$((stable + 1)); else stable=0; fi
            prev="$rss"
            cand_comm="$(cat "/proc/$cand/comm" 2>/dev/null || echo '?')"
            if [[ "$stable" -ge 2 || ( "$prov_every" -gt 0 && $((ticks_above % prov_every)) -eq 0 ) ]]; then
                if [[ "$cand_comm" != "$expect_comm" ]]; then
                    echo "WARNING: the largest resident process in our own tree is pid $cand" >&2
                    echo "         with comm '$cand_comm', not the expected '$expect_comm'." >&2
                    echo "         Sampling it anyway (it IS our descendant), but check the" >&2
                    echo "         proof below names the process you think it does." >&2
                fi
                attempts=$((attempts + 1))
                attempt_log="${PLACEMENT_LOG}.attempt-${attempts}"
                bash "$PLACEMENT_CHECK" "$cand" \
                    --threshold "$MAX_NODE_SHARE" \
                    --label "canonical:$(basename "$MODEL")" > "$attempt_log" 2>&1
                rc=$?
                cov="$(placement_coverage_pct "$attempt_log")"; last_cov="$cov"
                echo "$rc" > "${attempt_log}.rc"
                # accepted: the checker judged it (0 PASS/ADVISORY, 3 SKEW - skew IS a valid proof of placement)
                # AND the sample covers enough of the resident set. cov=-1 (no VmRSS line) is accepted.
                if { [[ "$rc" == "0" || "$rc" == "3" ]]; } && { [[ "$cov" == "-1" ]] || [[ "$cov" -ge "$min_cov" ]]; }; then
                    if [[ "$stable" -ge 2 ]]; then
                        cp "$attempt_log" "$PLACEMENT_LOG"; echo "$rc" > "${PLACEMENT_LOG}.rc"
                        printf 'captured: pid=%s comm=%s rss_pages=%s (%s GiB) expected_comm=%s settled=yes coverage_pct=%s attempt=%s\n' \
                            "$cand" "$cand_comm" "$rss" "$((rss / 262144))" "$expect_comm" "$cov" "$attempts" \
                            > "$PLACEMENT_REASON"
                        return 0
                    fi
                    # provisional: taken in-window but RSS still creeping; keep the latest, keep sampling
                    cp "$attempt_log" "${PLACEMENT_LOG}.provisional"; echo "$rc" > "${PLACEMENT_LOG}.provisional.rc"
                    prov_rc="$rc"; prov_pid="$cand"; prov_comm="$cand_comm"; prov_rss="$rss"; prov_cov="$cov"
                else
                    echo "NOTE: placement sample $attempts rejected (checker rc=$rc, coverage ${cov}% < ${min_cov}%); keeping $attempt_log, sampling again." >&2
                    stable=0
                fi
                if [[ "$attempts" -ge "$max_attempts" && -z "$prov_rc" ]]; then break; fi
            fi
        fi
        sleep "$interval"
    done

    # The run ended (or attempts ran out) without a settled sample. A provisional sample taken during the run is
    # still an in-window sample: promote the latest one, labelled.
    if [[ -n "$prov_rc" ]]; then
        cp "${PLACEMENT_LOG}.provisional" "$PLACEMENT_LOG"; echo "$prov_rc" > "${PLACEMENT_LOG}.rc"
        printf 'captured: pid=%s comm=%s rss_pages=%s (%s GiB) expected_comm=%s settled=no(provisional, RSS still changing) coverage_pct=%s attempt=%s\n' \
            "$prov_pid" "$prov_comm" "$prov_rss" "$((prov_rss / 262144))" "$expect_comm" "$prov_cov" "$attempts" \
            > "$PLACEMENT_REASON"
        return 0
    fi

    # LOUD failure: say WHY there is no proof, not merely that it is missing.
    {
        echo "NO in-window placement sample was captured."
        echo "  root pid                   : $root (exited before any sample qualified)"
        echo "  expected process comm      : $expect_comm"
        echo "  largest descendant RSS seen: ${best_seen} pages ($((best_seen / 262144)) GiB)"
        echo "  required floor             : ${floor_pages} pages"
        echo "  consecutive-stable samples : ${stable} (2 required)"
        echo "  checks attempted           : ${attempts} (last coverage ${last_cov}%, minimum ${min_cov}%)"
        if [[ "$best_seen" -lt "$floor_pages" ]]; then
            echo "  DIAGNOSIS: the descendant walk never saw a process above the floor."
            echo "             Either the run died during load, or the walk is broken again -"
            echo "             re-read the DS41-T6c note above largest_rss_descendant()."
        elif [[ "$attempts" -gt 0 ]]; then
            echo "  DIAGNOSIS: samples were taken but none was accepted (checker failed or the numastat"
            echo "             total covered too little of the resident set). See ${PLACEMENT_LOG}.attempt-N."
        else
            echo "  DIAGNOSIS: the run ended before the first check fired (run shorter than the load plus"
            echo "             the settle window). Lower PLACEMENT_SAMPLE_INTERVAL or lengthen the run."
        fi
        echo "  descendant tree at give-up time (may already be empty):"
        descendant_tree_dump "$root"
    } > "$PLACEMENT_REASON" 2>&1
    return 1
}
