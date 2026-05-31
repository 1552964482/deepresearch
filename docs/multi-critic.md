# Multi-Critic vs Single-Red Review

An iteration on the Phase-3 adversarial review. Instead of one Red agent
juggling four attack dimensions, **three persona-specialized critics** run in
parallel:

| Critic | Scope | Attack type |
|---|---|---|
| FactChecker | factual errors / unsupported claims | `factual` |
| LogicReviewer | internal contradictions / reasoning gaps | `logic` |
| CitationAuditor | missing / mismatched / unsourced citations | `citation` |

Their attacks are merged by a **two-signal consensus**:

1. **Span-level dedup** — near-duplicate spans (cosine ≥ 0.82 via bge-small-zh)
   collapse to their highest-severity representative.
2. **Section-level hot-spot** — since the personas cover disjoint dimensions
   they rarely hit the exact same span; a *section* flagged by ≥2 personas is
   treated as a problem hot-spot and up-weighted. Lone single-critic attacks
   are mildly down-weighted.

Design intent: **precision over recall** — each persona is narrowly scoped,
so its attacks are higher quality and more likely to survive Blue's
citation-preservation invariant.

## Results (ResearchBench, n=35, paired, BCa 95% CI)

All three rows use the same saved pipeline-r0 reports as input (zero extra
Tavily quota); only the reviewer differs.

| config | factual_acc | hallu_rate | cite_cov | judge_overall |
|---|---|---|---|---|
| pipeline-r0 (no review) | 1.000 | 0.164 | 0.393 | 4.299 |
| r2 single-Red | 1.000 | 0.158 | 0.472 | 4.329 |
| **r2 Multi-Critic** | 1.000 | 0.159 | **0.474** | **4.391** |

### Effect size vs pipeline-r0 (Cohen's d)

| metric | single-Red | Multi-Critic |
|---|---|---|
| citation_coverage | +1.31 (huge) | **+1.32 (huge)** |
| judge_overall | +0.06 (negligible) | **+0.19 (small)** |
| hallucination_rate | −0.06 | −0.05 |

## Interpretation

- **Citation coverage**: both reviewers tie (~0.47). The gain is driven by the
  citation-preservation invariant + CitationAuditor / Red's citation attacks,
  not by the multi-agent structure per se.
- **Judge overall**: Multi-Critic shows a ~3× larger effect on the holistic
  Judge score (d=+0.19 vs +0.06). The disjoint, focused personas surface
  higher-quality factual/logic issues that Blue can act on, which the holistic
  Judge rewards.
- **Honest caveat**: judge scores ran via the mimo fallback (aveve.xyz was down
  during these runs), so absolute judge magnitudes carry self-bias risk. The
  *relative* comparison (same judge for both arms) is still informative, and
  the citation_coverage metric is judge-independent.

## Reproduce (no Tavily needed)

```bash
# single-Red rerun
python experiments/rerun_red_blue.py \
  --src experiments/results/<TS>-ablation/pipeline-r0/reports \
  --reviewer red --review 2 \
  --compare-with experiments/results/<TS>-ablation/pipeline-r0/per_question.csv

# Multi-Critic rerun
python experiments/rerun_red_blue.py \
  --src experiments/results/<TS>-ablation/pipeline-r0/reports \
  --reviewer multi --review 2 \
  --compare-with experiments/results/<TS>-ablation/pipeline-r0/per_question.csv
```

Or live:

```bash
dr-agent run "你的研究问题" --review 2 --reviewer multi
```
