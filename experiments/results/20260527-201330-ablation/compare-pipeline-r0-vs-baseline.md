# Comparison: **pipeline-r0** vs **baseline**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 35 | 0.960 | 1.000 | +0.040 ⬆ better | +0.597 |
| hallucination_rate | 35 | 0.146 | 0.164 | +0.018 ⬆ worse | +0.141 |
| citation_coverage | 35 | 0.000 | 0.393 | +0.393 ⬆ better | +8.650 |
| judge_overall | 35 | 4.240 | 4.299 | +0.059 ⬆ better | +0.129 |
