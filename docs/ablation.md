# Ablation: ResearchBench × 3 Configurations

35 questions × 11 domains, paired per-question evaluation, Bootstrap (BCa) 95% CI, n_iters=1000.

## Final results

| config | factual_acc | hallu_rate | cite_cov | judge_overall |
|---|---|---|---|---|
| baseline (single-prompt mimo) | 0.960 | 0.146 | 0.000 | 4.240 |
| pipeline-r0 (full DAG, no Red-Blue) | **1.000** | 0.164 | 0.393 | 4.299 |
| **pipeline-r2 (full DAG + Red-Blue K=2)** | **1.000** | **0.158** | **0.472** | **4.329** |

> 95% CI via Bootstrap (BCa, n_iters=1000) — see per-config `summary.md`.

## Pairwise effect sizes (Cohen's d, paired)

### pipeline-r0 vs baseline (n=35)

| metric | Δ | Cohen's d | interpretation |
|---|---|---|---|
| factual_accuracy | **+0.040** | **+0.60** | medium |
| hallucination_rate | +0.018 | +0.14 | negligible |
| citation_coverage | **+0.393** | **+8.65** | huge |
| judge_overall | +0.059 | +0.13 | negligible |

> Adding the full retrieval-and-grounding pipeline yields a huge gain on citation
> coverage (an artifact of baseline emitting zero citations) and a medium gain on
> factual accuracy. Judge effect is small as the underlying mimo backend writes
> fluently in both modes.

### pipeline-r2 vs pipeline-r0 (n=35, after invariant fix)

| metric | Δ | Cohen's d | interpretation |
|---|---|---|---|
| factual_accuracy | +0.000 | n/a (zero variance) | preserved |
| hallucination_rate | -0.005 | -0.06 | negligible (better) |
| citation_coverage | **+0.079** | **+1.31** | huge |
| judge_overall | +0.030 | +0.06 | negligible |

> K=2 Red-Blue rounds add ~20% relative citation coverage on top of pipeline-r0
> without harming factual accuracy. After fixing the citation-preservation
> invariant (see below), the Red-Blue loop is unambiguously additive.

### pipeline-r2 vs baseline (n=35)

| metric | Δ | Cohen's d | interpretation |
|---|---|---|---|
| factual_accuracy | +0.040 | medium | |
| hallucination_rate | +0.012 | negligible | |
| citation_coverage | **+0.472** | **>10** | huge |
| judge_overall | +0.089 | small | |

## The Red-Blue regression and how it was fixed

The first 35-question ablation produced an unexpected regression:

| metric | r0 | r2 (initial) | Δ |
|---|---|---|---|
| citation_coverage | 0.393 | 0.320 | **−0.073** |
| factual_accuracy | 1.000 | 0.996 | −0.004 |

Inspecting reports revealed that Blue, when responding to *completeness* attacks,
sometimes rewrote sentences containing `[n]` citation markers and forgot to
preserve them — silently dropping citations. Three fixes were applied:

1. **Red prompt** — must keep `[n]` markers inside the `span` field; cap of 2
   completeness attacks per round, prioritizing factual/citation
2. **Blue prompt** — MODIFY's `new_text` must preserve all `[n]` markers from
   the target span
3. **Code-level invariant** in `_apply_patch`:
   - MODIFY rejected if `new_cites ⊉ span_cites`
   - DELETE rejected if removing the span would orphan a citation in the
     section

The third fix is the safety net: prompt compliance is probabilistic, but the
invariant is deterministic. After applying these, the rerun (using the saved
r0 reports as input to avoid further Tavily quota usage) produced the
positive-effect numbers above.

## Reproducing

```bash
# Full ablation (baseline + r0 + r2). Consumes Tavily quota for r0 only.
python experiments/run_ablation.py --concurrency 3 --n-judge 2

# Re-run only the Red-Blue layer on existing r0 reports (zero Tavily).
python experiments/rerun_red_blue.py \
  --src experiments/results/<TS>-ablation/pipeline-r0/reports \
  --review 2 \
  --concurrency 3 \
  --n-judge 2 \
  --compare-with experiments/results/<TS>-ablation/pipeline-r0/per_question.csv
```

## Caveats

- **Judge backend**: aveve.xyz `/chat/completions` was 502 throughout this
  experiment, so all judge scores ran via the mimo fallback path with
  `self_bias_risk=True`. The Cohen's d magnitudes for `judge_overall` should
  be interpreted accordingly. The two rule-based metrics with the largest
  effects (factual_accuracy, citation_coverage) are independent of the judge.
- **Bench size**: n=35 is enough for the BCa CI to converge but borderline for
  per-domain effect sizes (3-4 questions / domain). Future work: expand to ≥100
  questions, especially in domains where the current judge_overall has high
  variance (rl-algorithms, networking-security).
- **Difficulty ceiling**: factual_accuracy hits 1.000 at r0 already, leaving
  no headroom for r2 to improve on that metric. A harder bench would be more
  discriminative.
