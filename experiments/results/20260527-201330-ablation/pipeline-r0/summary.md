# Eval Summary — backend=mimo mode=pipeline

- samples evaluated: 35 (0 failed)
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.164 | [0.135, 0.198] |
| citation_coverage | 0.393 | [0.373, 0.414] |
| judge_overall (1-5) | 4.299 | [3.883, 4.413] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.169 | 0.413 | 4.400 |
| biomedical | 3 | 1.000 | 0.167 | 0.408 | 4.500 |
| data-science-stats | 3 | 1.000 | 0.243 | 0.366 | 4.340 |
| databases | 3 | 1.000 | 0.168 | 0.384 | 4.153 |
| economics-finance | 3 | 1.000 | 0.268 | 0.383 | 4.300 |
| history-geography | 4 | 1.000 | 0.153 | 0.394 | 4.165 |
| llm-rlhf | 4 | 1.000 | 0.081 | 0.404 | 4.455 |
| networking-security | 3 | 1.000 | 0.227 | 0.383 | 4.513 |
| rl-algorithms | 3 | 1.000 | 0.082 | 0.351 | 3.403 |
| software-engineering | 3 | 1.000 | 0.185 | 0.396 | 4.603 |
| systems-distributed | 3 | 1.000 | 0.090 | 0.437 | 4.447 |
