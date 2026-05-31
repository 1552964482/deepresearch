# Eval Summary — backend=mimo mode=baseline

- samples evaluated: 35 (0 failed)
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 0.960 | [0.916, 0.982] |
| hallucination_rate | 0.146 | [0.106, 0.205] |
| citation_coverage | 0.000 | [0.000, 0.000] |
| judge_overall (1-5) | 4.240 | [4.136, 4.342] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.140 | 0.000 | 4.370 |
| biomedical | 3 | 0.863 | 0.143 | 0.000 | 4.193 |
| data-science-stats | 3 | 1.000 | 0.271 | 0.000 | 4.493 |
| databases | 3 | 1.000 | 0.115 | 0.000 | 4.080 |
| economics-finance | 3 | 1.000 | 0.363 | 0.000 | 4.053 |
| history-geography | 4 | 0.893 | 0.177 | 0.000 | 3.928 |
| llm-rlhf | 4 | 1.000 | 0.019 | 0.000 | 4.453 |
| networking-security | 3 | 0.958 | 0.169 | 0.000 | 4.553 |
| rl-algorithms | 3 | 0.897 | 0.123 | 0.000 | 4.200 |
| software-engineering | 3 | 1.000 | 0.062 | 0.000 | 4.133 |
| systems-distributed | 3 | 0.958 | 0.053 | 0.000 | 4.220 |
