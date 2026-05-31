# Eval Summary — backend=mimo mode=pipeline-rerun

- samples evaluated: 35 (0 failed)
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback

## Aggregate metrics (95% CI via Bootstrap, BCa)

| metric | mean | 95% CI |
|---|---|---|
| factual_accuracy | 1.000 | [1.000, 1.000] |
| hallucination_rate | 0.158 | [0.125, 0.191] |
| citation_coverage | 0.472 | [0.454, 0.489] |
| judge_overall (1-5) | 4.329 | [4.121, 4.432] |

## By-domain breakdown

| domain | n | factual_acc | hallu | cite_cov | judge_overall |
|---|---|---|---|---|---|
| ai-ml-fundamentals | 3 | 1.000 | 0.151 | 0.520 | 4.333 |
| biomedical | 3 | 1.000 | 0.161 | 0.500 | 4.517 |
| data-science-stats | 3 | 1.000 | 0.260 | 0.445 | 4.553 |
| databases | 3 | 1.000 | 0.171 | 0.467 | 4.313 |
| economics-finance | 3 | 1.000 | 0.251 | 0.460 | 4.427 |
| history-geography | 4 | 1.000 | 0.159 | 0.444 | 4.207 |
| llm-rlhf | 4 | 1.000 | 0.076 | 0.483 | 4.540 |
| networking-security | 3 | 1.000 | 0.216 | 0.457 | 4.130 |
| rl-algorithms | 3 | 1.000 | 0.081 | 0.435 | 3.580 |
| software-engineering | 3 | 1.000 | 0.163 | 0.473 | 4.483 |
| systems-distributed | 3 | 1.000 | 0.079 | 0.513 | 4.503 |
