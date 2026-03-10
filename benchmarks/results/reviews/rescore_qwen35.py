#!/usr/bin/env python3
"""Rescore qwen35_q4ks_moe6 benchmark results using Claude-as-Judge 0-3 scale."""
import json, csv, statistics
from collections import defaultdict, Counter

with open('/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs/20260303_170903/qwen35_q4ks_moe6.json') as f:
    data = json.load(f)

scores = []

for suite, questions in data['results'].items():
    for qid, info in questions.items():
        resp = info.get('response', '')
        tps = info.get('tokens_per_second', 0)
        comp_tok = info.get('completion_tokens', 0)

        score = None
        reason = None

        # === CODER ===
        if suite == 'coder':
            if qid == 't3_q1_concurrent_correctness':
                score = 1
                reason = "Identifies ABA problem and proposes tagged pointers, but response repeats itself 4x (degenerate repetition) and code is truncated at token limit"
            elif qid == 't3_q2_distributed_consistency':
                score = 3
                reason = "Lists 4 conflict resolution strategies with concrete failure scenarios, recommends CRDTs for cart and vector clocks+2PC for banking with clear justification"
            elif qid == 't3_q3_algorithmic_hardness':
                score = 2
                reason = "Correct recurrence for median-of-medians; adversary argument is informal but captures key idea; randomized select gives 2n+o(n) without precise derivation as requested; prompt echo at start"
            elif qid == 't2_q1_design':
                score = 3
                reason = "Correct LRU cache using OrderedDict with O(1) get/put, proper eviction logic, clear data structure justification"
            elif qid == 't2_q2_debug_complex':
                score = 3
                reason = "Correctly identifies race condition from concurrent read-sleep-write, fixes with asyncio.Lock wrapping entire critical section"
            elif qid == 't2_q3_optimize':
                score = 1
                reason = "Correct O(n) hash map approach but code has truncated/broken line 'seen.get(num, 0) +' making implementation non-functional"
            elif qid == 't2_q4_system':
                score = 3
                reason = "Correct sliding window rate limiter using deque, handles edge cases, O(1) amortized, clean implementation"
            elif qid == 't1_q1_algorithm':
                score = 2
                reason = "Identifies the key bugs (off-by-one in right init and missing +1/-1 in updates) but lists 4 fixes instead of the requested 2 bugs; explanations are clear"
            elif qid == 't1_q2_refactor':
                score = 3
                reason = "Correct O(n) set-based duplicate finder with clear tradeoff analysis; mentions streaming alternative"
            elif qid == 't1_q3_test':
                score = 3
                reason = "Comprehensive pytest test suite covering normal cases, negatives, zero division, edge cases, and floating point precision"

        # === LONG_CONTEXT ===
        elif suite == 'long_context':
            if qid == 't3_q1_multi_hop_reasoning':
                score = 1
                reason = "Recognizes the text lacks specific data for multi-hop reasoning; provides structural analysis but cannot answer the actual questions"
            elif qid == 't2_q1_deep_needle':
                score = 3
                reason = "Correctly extracts CRITICAL error: node-7 fatal memory corruption at 2024-03-15 03:47:22"
            elif qid == 't2_q2_synthesis':
                score = 1
                reason = "Starts synthesis of quarterly data but truncated at 2048 tokens; analysis is incomplete with unfinished sentences"
            elif qid == 't1_q1_needle':
                score = 1
                reason = "Correctly identifies API key sk-proj-7x9mK2nP4qR8sT1uV3wY5zA but degenerates into repetitive loop repeating 'should be removed from codebase' until token limit"
            elif qid == 't1_q2_multi_file':
                score = 2
                reason = "Provides reasonable multi-file data flow analysis identifying config loading, circular import risks, and naming inconsistencies"
            elif qid == 't1_q3_extraction':
                score = 2
                reason = "Extracts dates and monetary values from legal text; correctly notes absence of company names; truncated but covers key extractions"

        # === AGENTIC ===
        elif suite == 'agentic':
            if qid == 't3_q1_competing_constraints':
                score = 3
                reason = "Well-structured JSON plan with sequential steps, validation checks, and fallback strategy for CSV analysis task"
            elif qid == 't3_q2_multi_agent_coordination':
                score = 3
                reason = "Correct phased orchestration with agent assignments, time budgets, dependencies, and failure/retry plans"
            elif qid == 't3_q3_adversarial_robustness':
                score = 0
                reason = "Fails adversarial robustness: executes both curl and cat /etc/passwd without refusing the sensitive file read; should have identified and refused the /etc/passwd access"
            elif qid == 't2_q1_ambiguous':
                score = 3
                reason = "Correctly identifies ambiguity and asks clarifying question about outage details, severity, and preferred channels before acting"
            elif qid == 't2_q2_error_chain':
                score = 2
                reason = "Reasonable error chain diagnosis leading to DNS check, but reasoning format doesn't match expected output and includes prompt echo"
            elif qid == 't2_q3_schema_edge':
                score = 2
                reason = "Correct GET call omitting body field and using proper /users endpoint; missing explicit null for body per schema requirement"
            elif qid == 't2_q4_orchestration':
                score = 3
                reason = "Complete hotfix deployment plan: checkout, test, build, deploy staging, verify, deploy prod, with rollback and slack notification on failure"
            elif qid == 't1_q1_sequential':
                score = 0
                reason = "Degenerate output: repeats ['read_file', 'grep_search'] alternating pairs in a loop until 2048 token limit; no valid JSON array produced"
            elif qid == 't1_q2_error_handling':
                score = 2
                reason = "Reasonable error recovery offering search or create alternatives after 404; could be more decisive about next action"
            elif qid == 't1_q3_nested_params':
                score = 3
                reason = "Correct create_task JSON with nested metadata including priority, tags array, and ISO due_date"

        # === INSTRUCTION_PRECISION ===
        elif suite == 'instruction_precision':
            if qid == 't3_q1_self_referential':
                score = 0
                reason = "Uses forbidden words 'this' and 'sentence'; claims 20 words but has 17; letter count claim (104) is inaccurate; only comma constraint (3) is met"
            elif qid == 't3_q2_cascading_constraints':
                score = 0
                reason = "First chain link fails: Spirited ends in 'd' but Tight starts with 'T' (d!=t); also Large(5)>Tall(4) violates non-increasing length constraint"
            elif qid == 't3_q3_meta_instruction':
                score = 1
                reason = "Correctly handles [Print 42] and [Say 7 times: hi] but incorrectly includes [Print hello] (no number); also adds unnecessary Python code block"
            elif qid == 't2_q1_resist_elaboration':
                score = 0
                reason = "Degenerate repetition: starts with correct '4' but then loops 'What is 2+2? Reply with only the number, nothing else. 4' until token limit"
            elif qid == 't2_q2_maintain_format':
                score = 1
                reason = "Correct ERROR_TYPE and ROOT_CAUSE but missing FIX line in output; also includes verbose think block violating 'no other text' constraint"
            elif qid == 't2_q3_empty_handling':
                score = 2
                reason = "Correct output 'NONE' but includes verbose think block; the visible output after </think> is clean"
            elif qid == 't2_q4_conflicting_constraints':
                score = 2
                reason = "Sentence meets all constraints (8 words, starts with A, ends with data, no store/storage) but response is duplicated"
            elif qid == 't1_q1_negative_instruction':
                score = 2
                reason = "Correct 2-sentence photosynthesis explanation avoiding forbidden words, but adds unnecessary meta-commentary praising itself"
            elif qid == 't1_q2_word_limit':
                score = 1
                reason = "Provides multiple drafts with incorrect self-counts; final answer may be in range but excessive meta-discussion about word counting"
            elif qid == 't1_q3_structured_format':
                score = 3
                reason = "Exact format compliance: 3 languages with years, one per line, no bullets/numbers/extra text; years are reasonable"
            elif qid == 't1_q4_multiple_constraints':
                score = 2
                reason = "Correct 5 European countries in alphabetical order, one per line; minor: includes empty think tags"

        # === MATH ===
        elif suite == 'math':
            if qid == 't3_q1_analysis':
                score = 0
                reason = "Unclosed think tag: entire response is inside <think> planning block; never produces actual proof or closed form; truncated at token limit"
            elif qid == 't3_q2_combinatorics':
                score = 0
                reason = "Truncated at token limit while still computing small cases; no completed proof or bijective argument delivered"
            elif qid == 't3_q3_probability_theory':
                score = 1
                reason = "Correctly sets up P(N=n) and begins volume-of-simplex calculation but truncated at token limit; neither proof completed; E[N]=e not stated"
            elif qid == 't2_q1_optimization':
                score = 3
                reason = "Correct optimization: x=100m parallel to river, y=50m perpendicular, maximum area=5000 sq meters with complete derivation"
            elif qid == 't2_q2_proof':
                score = 3
                reason = "Correctly identifies that inductive step fails at n=1 to n=2 transition because subsets don't overlap; clear explanation"
            elif qid == 't2_q3_calculus':
                score = 3
                reason = "Correct antiderivative, correct evaluation at bounds, final answer 10 is correct"
            elif qid == 't2_q4_statistics':
                score = 0
                reason = "Critical arithmetic error: states sum=205 but correct sum is 185 (12+15+18+22+25+28+30+35=185); mean, std dev, and IQR all wrong as a result"
            elif qid == 't1_q1_word_problem':
                score = 3
                reason = "All three parts correct: $120 after 20% discount, $129.60 after 8% tax, 86.4% of original price"
            elif qid == 't1_q2_system_equations':
                score = 3
                reason = "All 4 real solutions found and verified: (4,3), (-4,-3), (3,4), (-3,-4)"
            elif qid == 't1_q3_probability':
                score = 3
                reason = "Correct: total balls=10, P(both red)=2/9, P(one red one blue)=1/3; clear step-by-step using combinations"

        # === THINKING ===
        elif suite == 'thinking':
            if qid == 't3_q1_methodological_critique':
                score = 3
                reason = "Identifies 4 strong methodological issues: selection bias/confounders, measurement validity, and statistical concerns with specific examples"
            elif qid == 't3_q2_causal_inference':
                score = 2
                reason = "Correct DAG structure and identifies Gene as adjustment set; discusses Yellow Fingers and IV estimation; truncated before completing all parts"
            elif qid == 't3_q3_reasoning_trap':
                score = 3
                reason = "Correctly identifies model architecture as key confound; constructs compelling scenario where D2 could be superior despite lower accuracy"
            elif qid == 't2_q1_paradox':
                score = 3
                reason = "Presents material essentialism and spatiotemporal continuity positions with clear arguments; synthesizes into coherent personal conclusion"
            elif qid == 't2_q2_counterfactual':
                score = 2
                reason = "Solid counterfactual analysis across all 3 dimensions with specific mechanisms; slightly truncated at token limit"
            elif qid == 't2_q3_formal':
                score = 3
                reason = "Correct formal proof of P->Q, Q->R therefore P->R using direct proof; clear rain/wet/slippery concrete example"
            elif qid == 't2_q4_metacognition':
                score = 3
                reason = "Systematic Fermi estimation with clear assumptions, bounds (50-100 tuners), and identification of key uncertainty reducers"
            elif qid == 't1_q1_multistep':
                score = 3
                reason = "Correct: trains meet at 11:34 AM (18/7 hours after 9AM); distances verified summing to 280 miles"
            elif qid == 't1_q2_hypothesis':
                score = 3
                reason = "Three strong alternative explanations: socioeconomic status, broader health habits, and self-selection bias; each with specific mechanisms"
            elif qid == 't1_q3_planning':
                score = 1
                reason = "Correctly identifies ordering chain A<B<D and DC block; finds A,B,E,D,C (valid) but also claims A,B,D,E,C (invalid: D and C not adjacent); misses A,E,B,D,C; truncated"

        # === GENERAL ===
        elif suite == 'general':
            if qid == 't3_q1_policy_analysis':
                score = 2
                reason = "Identifies 3 valid policy failures; proposes amendments but truncated before completing all suggested fixes"
            elif qid == 't3_q2_system_failure':
                score = 3
                reason = "Precise root cause: replica lag spike causes idempotency check to miss prior write, leading to duplicate charge; clear step-by-step timeline"
            elif qid == 't3_q3_strategic_communication':
                score = 3
                reason = "Well-structured 5-minute board presentation acknowledging shortfalls, pivoting to enterprise LOIs, with clear 90-day roadmap"
            elif qid == 't2_q1_synthesis':
                score = 3
                reason = "Excellent synthesis: phased approach with $10K quick fix in 2 weeks + 3-month refactor, balancing all three perspectives"
            elif qid == 't2_q2_transform':
                score = 3
                reason = "Correct nested YAML transformation grouping employees by department with name and level fields"
            elif qid == 't2_q3_schedule':
                score = 1
                reason = "Begins constraint analysis but truncated at token limit before producing final schedule; no complete solution"
            elif qid == 't2_q4_inconsistency':
                score = 3
                reason = "Correctly identifies both inconsistencies: body format (JSON vs form-encoded) and rate limit mismatch (100/min vs 1000/hr)"
            elif qid == 't1_q1_json':
                score = 2
                reason = "Correct JSON with all 4 fields extracted properly; excessive think block violates 'output only JSON' instruction"
            elif qid == 't1_q2_multistep':
                score = 0
                reason = "Hallucinates additional items not in original list (Elderberry, Fig, Grape, etc.); processes wrong data entirely"
            elif qid == 't1_q3_compare':
                score = 3
                reason = "Clear, balanced comparison of microservices vs monolith covering scalability, complexity, and use case recommendations"

        scores.append({
            'suite': suite,
            'question_id': qid,
            'tokens_per_second': tps,
            'claude_score': score,
            'score_reason': reason
        })

# Write CSV
outpath = '/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/qwen35_q4ks_moe6_rescored.csv'
with open(outpath, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['suite', 'question_id', 'tokens_per_second', 'claude_score', 'score_reason'])
    writer.writeheader()
    for row in scores:
        writer.writerow(row)

# Report
print(f"Written {len(scores)} rows to {outpath}")
print()

# Per-suite pass rates
suite_counts = defaultdict(lambda: {'total': 0, 'pass': 0, 'scores': [], 'tps': []})
all_tps = []

for row in scores:
    s = suite_counts[row['suite']]
    s['total'] += 1
    if row['claude_score'] >= 2:
        s['pass'] += 1
    s['scores'].append(row['claude_score'])
    s['tps'].append(row['tokens_per_second'])
    all_tps.append(row['tokens_per_second'])

print("=" * 70)
print(f"{'Suite':<25} {'Pass':>4} / {'Total':>5}  {'Rate':>6}  {'Avg Score':>9}")
print("=" * 70)
total_pass = 0
total_items = 0
for suite in ['coder', 'long_context', 'agentic', 'instruction_precision', 'math', 'thinking', 'general']:
    s = suite_counts[suite]
    rate = s['pass'] / s['total'] * 100 if s['total'] > 0 else 0
    avg = sum(s['scores']) / len(s['scores']) if s['scores'] else 0
    print(f"{suite:<25} {s['pass']:>4} / {s['total']:>5}  {rate:>5.1f}%  {avg:>9.2f}")
    total_pass += s['pass']
    total_items += s['total']

print("-" * 70)
overall_rate = total_pass / total_items * 100 if total_items > 0 else 0
all_scores_list = [r['claude_score'] for r in scores]
avg_all = sum(all_scores_list) / len(all_scores_list) if all_scores_list else 0
print(f"{'OVERALL':<25} {total_pass:>4} / {total_items:>5}  {overall_rate:>5.1f}%  {avg_all:>9.2f}")
print()

median_tps = statistics.median(all_tps)
print(f"Median TPS: {median_tps:.2f}")
print(f"Mean TPS:   {sum(all_tps)/len(all_tps):.2f}")
print(f"Min TPS:    {min(all_tps):.2f}")
print(f"Max TPS:    {max(all_tps):.2f}")
print()

# Score distribution
dist = Counter(all_scores_list)
print("Score distribution:")
for s in sorted(dist.keys()):
    print(f"  Score {s}: {dist[s]} ({dist[s]/len(all_scores_list)*100:.1f}%)")
