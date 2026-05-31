# Eval Summary — backend=mimo mode=baseline

- samples evaluated: 3 (0 failed)
- judge backends seen: mimo-fallback×3
- ⚠️  self-bias risk: 3 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 0.944 | [0.833, 0.944] |
| hallucination_rate | 0.107 | [0.037, 0.162] |
| citation_coverage | 0.000 | [0.000, 0.000] |
| judge_overall (1-5) | 4.727 | [4.450, 4.877] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 1 | 1.000 | 0.204 | 0.000 | 4.830 |
| rl-algorithms | 1 | 0.833 | 0.037 | 0.000 | 4.450 |
| software-engineering | 1 | 1.000 | 0.078 | 0.000 | 4.900 |
