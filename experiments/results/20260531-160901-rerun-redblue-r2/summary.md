# Eval Summary — backend=mimo mode=pipeline-rerun

- samples evaluated: 2 (0 failed)
- judge backends seen: mimo-fallback×2
- ⚠️  self-bias risk: 2 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.178 | [0.168, 0.178] |
| citation_coverage | 0.514 | [0.453, 0.514] |
| judge_overall (1-5) | 4.530 | [4.250, 4.530] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 2 | 1.000 | 0.178 | 0.514 | 4.530 |
