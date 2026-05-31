# Eval Summary — backend=mimo mode=baseline

- samples evaluated: 2 (1 failed)
- judge backends seen: mimo-fallback×2
- ⚠️  self-bias risk: 2 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.160 | [0.070, 0.160] |
| citation_coverage | 0.000 | [0.000, 0.000] |
| judge_overall (1-5) | 4.395 | [4.130, 4.395] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 1 | 1.000 | 0.250 | 0.000 | 4.660 |
| software-engineering | 1 | 1.000 | 0.070 | 0.000 | 4.130 |
