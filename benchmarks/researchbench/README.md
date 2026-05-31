# ResearchBench (35 questions × 11 domains)

A small but diverse research-question benchmark used to evaluate the
DeepResearch-MultiAgent pipeline.

Each row in `questions.jsonl` has:

| field | type | meaning |
|---|---|---|
| `id` | str | stable identifier `rb-NNN` |
| `domain` | str | one of 11 domains (see below) |
| `question` | str | the research query fed to the agent |
| `reference_facts` | list[str] | atomic facts the report should mention; used by the **factual_accuracy** metric (embedding ≥ 0.55 = hit) |
| `forbidden_claims` | list[str] | known-false statements; if a report semantically matches one, it counts as a **hallucination** |
| `scoring_rubric` | list[str] | specific points appended to the Judge prompt for this question |
| `language` | str | `zh` / `en` |

## Domains (counts)

```
ai-ml-fundamentals     3
llm-rlhf               4
rl-algorithms          3
systems-distributed    3
software-engineering   3
databases              3
networking-security    3
data-science-stats     3
biomedical             3
economics-finance      3
history-geography      4
```

Total: **35**.
