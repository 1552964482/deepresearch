# Eval Summary — backend=mimo mode=pipeline

- samples evaluated: 35 (0 failed)
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 0.996 | [0.980, 1.000] |
| hallucination_rate | 0.156 | [0.127, 0.188] |
| citation_coverage | 0.320 | [0.252, 0.369] |
| judge_overall (1-5) | 4.285 | [4.182, 4.378] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.180 | 0.427 | 4.493 |
| biomedical | 3 | 1.000 | 0.171 | 0.438 | 4.367 |
| data-science-stats | 3 | 1.000 | 0.274 | 0.392 | 4.303 |
| databases | 3 | 1.000 | 0.172 | 0.351 | 4.163 |
| economics-finance | 3 | 1.000 | 0.180 | 0.000 | 4.113 |
| history-geography | 4 | 0.964 | 0.081 | 0.000 | 3.795 |
| llm-rlhf | 4 | 1.000 | 0.097 | 0.424 | 4.475 |
| networking-security | 3 | 1.000 | 0.226 | 0.390 | 4.397 |
| rl-algorithms | 3 | 1.000 | 0.097 | 0.385 | 4.233 |
| software-engineering | 3 | 1.000 | 0.191 | 0.380 | 4.503 |
| systems-distributed | 3 | 1.000 | 0.090 | 0.407 | 4.387 |
