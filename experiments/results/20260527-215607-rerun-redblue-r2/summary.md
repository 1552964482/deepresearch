# Eval Summary — backend=mimo mode=pipeline-rerun

- samples evaluated: 3 (0 failed)
- judge backends seen: mimo-fallback×3
- ⚠️  self-bias risk: 3 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.153 | [0.102, 0.183] |
| citation_coverage | 0.521 | [0.453, 0.558] |
| judge_overall (1-5) | 4.420 | [4.240, 4.610] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.153 | 0.521 | 4.420 |
