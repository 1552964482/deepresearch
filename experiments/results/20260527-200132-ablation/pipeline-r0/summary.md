# Eval Summary — backend=mimo mode=pipeline

- samples evaluated: 3 (0 failed)
- judge backends seen: mimo-fallback×3
- ⚠️  self-bias risk: 3 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.109 | [0.003, 0.178] |
| citation_coverage | 0.356 | [0.351, 0.359] |
| judge_overall (1-5) | 4.790 | [4.650, 4.863] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 1 | 1.000 | 0.209 | 0.359 | 4.650 |
| rl-algorithms | 1 | 1.000 | 0.003 | 0.351 | 4.850 |
| software-engineering | 1 | 1.000 | 0.115 | 0.357 | 4.870 |
