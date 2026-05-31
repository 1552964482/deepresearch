# Eval Summary — backend=mimo mode=pipeline

- samples evaluated: 3 (0 failed)
- judge backends seen: mimo-fallback×3
- ⚠️  self-bias risk: 3 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.098 | [0.008, 0.169] |
| citation_coverage | 0.345 | [0.290, 0.376] |
| judge_overall (1-5) | 4.543 | [4.450, 4.593] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 1 | 1.000 | 0.219 | 0.365 | 4.600 |
| rl-algorithms | 1 | 1.000 | 0.008 | 0.290 | 4.580 |
| software-engineering | 1 | 1.000 | 0.068 | 0.382 | 4.450 |
