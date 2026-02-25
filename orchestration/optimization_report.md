# Orchestrator Optimization Report

Generated: 2026-01-07T23:14:00.054275

## Summary

- Started: 2026-01-07T20:48:32.061707
- Completed: 2026-01-07T23:14:00.047514

## Layer Results

| Layer | Status | Trials | Score | Parameters |
|-------|--------|--------|-------|------------|
| frontdoor | complete | 5 | 1.0000 | temperature=0.074, speculative_k=31, entropy_threshold=4.090 |
| formalizer | complete | 5 | 0.5000 | temperature=0.190 |
| specialists | complete | 5 | 0.3000 | temperature=0.339, speculative_k=10, early_abort_tokens=65 |
| workers | complete | 5 | 1.0000 | temperature=0.493, repetition_threshold=0.270 |

## Optimal Configuration

```yaml
optimized_params:
  frontdoor:
    temperature: 0.0736
    speculative_k: 31
    entropy_threshold: 4.0905
  formalizer:
    temperature: 0.1901
  specialists:
    temperature: 0.3395
    speculative_k: 10
    early_abort_tokens: 65
  workers:
    temperature: 0.4928
    repetition_threshold: 0.2697
```

## Metrics by Layer

### frontdoor

- parse_success_rate: 1.0000
- task_completion_rate: 0.0000
- avg_turns: 1.0000
- latency_ms: 6.7974
- schema_validation_rate: 0.0000
- execution_success_rate: 1.0000
- escalation_accuracy: 0.0000

### formalizer

- parse_success_rate: 1.0000
- task_completion_rate: 0.0000
- avg_turns: 1.0000
- latency_ms: 4.8243
- schema_validation_rate: 0.0000
- execution_success_rate: 1.0000
- escalation_accuracy: 0.0000

### specialists

- parse_success_rate: 1.0000
- task_completion_rate: 0.0000
- avg_turns: 1.0000
- latency_ms: 3.1601
- schema_validation_rate: 0.0000
- execution_success_rate: 1.0000
- escalation_accuracy: 0.0000

### workers

- parse_success_rate: 1.0000
- task_completion_rate: 0.0000
- avg_turns: 1.0000
- latency_ms: 6.3312
- schema_validation_rate: 0.0000
- execution_success_rate: 1.0000
- escalation_accuracy: 0.0000

## Optimization History

- **frontdoor** completed 2026-01-07T20:48:35.761241
  - Trials: 5, Score: 1.0000
- **formalizer** completed 2026-01-07T21:20:52.524050
  - Trials: 5, Score: 0.5000
- **specialists** completed 2026-01-07T23:13:43.488155
  - Trials: 5, Score: 0.3000
- **workers** completed 2026-01-07T23:14:00.046830
  - Trials: 5, Score: 1.0000
