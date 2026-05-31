# Eval Summary — backend=mimo mode=pipeline

- samples evaluated: 3 (0 failed)
- judge backends seen: mimo-fallback×3
- ⚠️  self-bias risk: 3 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.117 | [0.003, 0.179] |
| citation_coverage | 0.335 | [0.269, 0.371] |
| judge_overall (1-5) | 4.577 | [4.500, 4.617] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 1 | 1.000 | 0.192 | 0.376 | 4.610 |
| rl-algorithms | 1 | 1.000 | 0.003 | 0.360 | 4.500 |
| software-engineering | 1 | 1.000 | 0.154 | 0.269 | 4.620 |
