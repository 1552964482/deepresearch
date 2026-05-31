# Eval Summary — backend=mimo mode=pipeline-rerun

- samples evaluated: 35 (0 failed)
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.159 | [0.127, 0.194] |
| citation_coverage | 0.474 | [0.455, 0.492] |
| judge_overall (1-5) | 4.391 | [4.153, 4.488] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.155 | 0.527 | 4.593 |
| biomedical | 3 | 1.000 | 0.165 | 0.495 | 4.557 |
| data-science-stats | 3 | 1.000 | 0.259 | 0.449 | 4.487 |
| databases | 3 | 1.000 | 0.169 | 0.466 | 4.467 |
| economics-finance | 3 | 1.000 | 0.247 | 0.457 | 4.617 |
| history-geography | 4 | 1.000 | 0.162 | 0.449 | 4.135 |
| llm-rlhf | 4 | 1.000 | 0.077 | 0.486 | 4.440 |
| networking-security | 3 | 1.000 | 0.218 | 0.461 | 4.280 |
| rl-algorithms | 3 | 1.000 | 0.081 | 0.435 | 3.753 |
| software-engineering | 3 | 1.000 | 0.165 | 0.475 | 4.600 |
| systems-distributed | 3 | 1.000 | 0.078 | 0.514 | 4.447 |
