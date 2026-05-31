# 在弱基座上把研究报告做对：DeepResearch-MultiAgent 的工程取舍

> 把 mimo-v2.5-pro 这种弱基座，通过工程化 Agent 编排，做成可以打过单点强模型的研究报告系统。本文记录 6 周从设计到落地到消融评测的关键取舍与翻车现场。

## 起点：要解决什么

LLM 直答一个研究问题（比如"GRPO 算法相对 PPO 的核心改进"）通常会出现三个问题：

1. **没有引用** — 一段技术性结论后面不带来源，可信度差
2. **覆盖不全** — 强模型也会漏掉关键侧面，比如忘了讨论 KL 惩罚
3. **难以量化** — 怎么"客观"评测一篇研究报告的质量

这个项目要做的就是把这三件事都解决，并且能拿出**可重复、有统计显著性**的实验数据。

## 架构选型：自研而不是 LangGraph

第一个非显然的决定是**不用 LangGraph**。LangGraph 一上来就解决了 DAG 编排和状态管理，但代价是：

- 一层不可控的抽象，调试时多一层栈
- 节点失败的传播策略不容易调
- 简历上"用了 LangGraph"不如"自研 DAG 调度器 + 9 状态状态机"有讲点

我用 `asyncio` + 一个表驱动的状态机自己实现：

```python
TRANSITIONS: dict[tuple[AgentState, str], AgentState] = {
    (S.IDLE, "start"): S.PLANNING,
    (S.PLANNING, "plan_ok"): S.SEARCHING,
    (S.SEARCHING, "search_ok"): S.READING,
    # ...
}
```

所有合法迁移在一个 dict 里，转移日志带时间戳全部记录，可以一行 `export_mermaid()` 出 mermaid 图给面试官看。

## LLM 网关：4 个 key 怎么用好

mimo 给了 4 个 API key，每个 100 RPM、安全并发 4。最朴素的做法是 round-robin 选 key，但这在突发流量下会把单 key 打爆触发 429。

我写的 **MimoPool** 做了三件事：

**1. 每 key 独立的并发槽 + 滑窗令牌桶**

每个 key 有：
- `asyncio.Semaphore(4)` — 硬并发上限
- `SlidingWindowBucket(rpm=100)` — 60 秒滑动窗口，按时间戳队列实现

请求要同时占住两个槽位才能发出去。

**2. Least-in-flight 选 key**

每次发请求按 `(in_flight ASC, recent_rpm_usage ASC)` 排序，挑当前最闲的 key。这避免了短时所有请求扎堆同一 key。

**3. 三段重试 + 指数退避**

- 同 key 重试 1 次（处理偶发网络抖动）
- 切到另一个可用 key 重试 1 次
- 全部失败抛 `LLMUnavailable`

429 时被 ban 的 key 会进入 `cooldown_until`，期间不参与选 key 排序。

实测 35 题完整 pipeline 跑下来，最高 RPM 用量也只有 1-3/100 —— 瓶颈实际上在单次 LLM 调用的延迟（mimo 平均 20-40s），4 keys × 4 并发 = 16 路对当前任务规模绰绰有余。

## 评测的合法性问题：Judge 必须独立

我看过一些项目用 mimo 自己当 Judge 评测自己 —— 这是实验设计错误。Zheng 等 2023 年 *Judging LLM-as-a-Judge* 和 Panickssery 等 2024 *LLMs Cannot Self-Reward* 都已经指出 LLM 系统性地偏好自己的输出风格。

我把 Judge 走完全独立的端点（aveve.xyz 上的 GPT-5.4），从 model family 到 base_url 都和被测分开。但中途遇到一个真实工程问题：

**aveve.xyz 中途 502 了**

evaluation 跑了一半发现 Judge 端点全部 `502 Bad Gateway`，三次重试都挂。这时面对两个选择：

- A. 等服务恢复（不可控）
- B. 降级到 mimo 当 Judge，但**清楚标注 self-bias 风险**

我选了 B：在 `JudgeClient` 里加 `fallback_pool` 参数，降级时给每条 score 打 `backend="mimo-fallback"` + `self_bias_risk=True`。Eval summary 里在头部明确写：

```
- judge backends seen: mimo-fallback×35
- ⚠️  self-bias risk: 35 samples scored via mimo fallback
```

这样数据还是能用、能继续开发，但**不会被误读**为独立 Judge 的结果。简历上这种"承认局限"反而是加分项。

## Red-Blue 对抗：第一次跑出来反而变差

最有意思的翻车现场。Phase 3 我设计了 Red-Blue 对抗循环：Red 从 4 维度（factual / logic / citation / completeness）找问题，Blue 出 patch 修复。预期 K=2 轮后报告质量提升。

第一次 35 题完整消融跑出来：

| 指标 | pipeline-r0 | pipeline-r2 | Δ |
|---|---|---|---|
| factual_accuracy | 1.000 | 0.996 | -0.004 |
| **citation_coverage** | **0.393** | **0.320** | **-0.073** |

引用覆盖率 r2 反而比 r0 低。怎么回事。

打开几份 r2 报告肉眼比对发现：

> **r0**：「GRPO 由 DeepSeek 提出 [1][2]，核心创新是组内相对优势 [3]」
> **r2**：「GRPO 由 DeepSeek 提出，是一种针对长链推理优化的算法（详见后文实验数据），核心创新是利用同一 prompt 下多次采样的相对排序」

Blue 在响应 completeness 攻击时**改写了带 `[1][2]` 的句子**，新句子里没保留引用编号，cite_cov 自然掉了。

修复策略三层：

1. **Red prompt** 改成"span 中含 `[n]` 的必须原样保留在 span 字段中"，并限制 completeness 攻击 ≤ 2 条
2. **Blue prompt** 强制要求 MODIFY 时 new_text 必须保留所有原 span 中的引用标记
3. **代码层 invariant**（最后一道关）：

```python
def _apply_patch(section: Section, patch: Patch):
    if patch.action is PatchAction.MODIFY:
        span_cites = _citations_in(span)
        new_cites = _citations_in(patch.new_text or "")
        if not span_cites.issubset(new_cites):
            return section, False  # reject patch
```

这里关键点是**不只信 prompt**。LLM 的 prompt 遵循能力是概率事件，工程上必须有最后一道代码 invariant 兜底。同样的，DELETE 操作如果会让 `[n]` 在该 section 中孤立（没有其他句子引用同一个编号），patch 也会被拒绝。

加这个 invariant 后 3 题 smoke test：

| 指标 | r0 | r2 (修复后) | Δ |
|---|---|---|---|
| citation_coverage | 0.413 | **0.521** | **+0.108** |
| hallucination_rate | 0.169 | 0.153 | -0.016 |

方向回正了。

> 📌 完整 35 题 paired 数据见 `experiments/results/<TIMESTAMP>-rerun-redblue-r2/compare-vs-r0.md`。

## JSON 解析的三层 fallback

LLM 输出 JSON 不是确定性的。即使你显式设了 `response_format={"type":"json_object"}`，偶尔还是会被 markdown fence 或注释污染。我设计的三层 fallback：

1. **direct**：原始响应直接 `json.loads`，外加去 markdown fence + 正则提取 `{...}` 块兜底
2. **strict-retry**：把上次错误响应作为 history 注入，再发一条系统消息 "请只输出 JSON，无解释"，温度降到 0
3. **regex**：从所有历次 raw 响应中正则抓 `{...}` 块，能解析出来就用

实测 35 题 × K=2 = 70 次 Red 调用里大部分走 `direct`，偶尔触发 `retry-strict`，从未需要 `regex`。但**触发过**就是简历卖点：

> "设计 direct → strict-JSON-retry → 正则提取的三层 JSON 解析 fallback。在 70+ 次真实调用中触发 retry-strict 修复 N 次，从未失败到第三层。"

## 三级语义压缩

研究报告的 Reader 阶段需要把搜回来的 chunks（每篇 1.5k 字符 × 5 篇 × 6 SubTasks）塞给 Writer。直接拼起来 token 爆了。

三级压缩是工业界常见做法的简化版：

- **L1 Embedding 过滤**：bge-small-zh 把 chunk 和 SubTask query 编码成 512-D 向量，cosine < 0.45 直接丢
- **L2 TextRank**：保留下来的 chunk 内做句子级 TextRank，jieba 分词后构图，PageRank 排序保留 top-8
- **L3 命名实体 / 数字 / 引用原句保留**：被 L2 排除的句子如果含 `\d+`、`\d{4}\s*年`、引号内片段、大写缩写，**强制保留**

L3 是最容易翻车的地方 —— 老实说一开始单元测试错误地构造了"含数字的非保护句子"，跑 Compressor 后预期被 budget 截掉，结果它被 L3 当成保护句子留了下来，测试挂了。这其实是测试构造错了，反而验证了 L3 工作正常。

## 评测细节：为什么是 BCa 而不是 percentile bootstrap

Bootstrap 95% CI 有两种主流做法：

- **Percentile**：直接取重采样统计量分布的 2.5% / 97.5% 分位数，最简单
- **BCa (Bias-Corrected and accelerated)**：加偏差校正 z₀ + 加速因子 a，对小样本和偏态分布更准

n=35 不算大样本，分布也未必对称（factual_accuracy 接近天花板时左偏），我选 BCa。代码上 30 行 numpy + scipy 搞定，jackknife 算 acceleration。

```python
# Bias correction
z0 = scipy_stats.norm.ppf((samples < point).mean())

# Acceleration via jackknife
jack = [statistic_fn(np.delete(data, i)) for i in range(n)]
a = ((jack.mean() - jack)**3).sum() / (6 * ((jack.mean() - jack)**2).sum()**1.5)

alpha1 = norm.cdf(z0 + (z0 + z_α/2) / (1 - a*(z0 + z_α/2)))
```

跑出来的 CI 比 percentile 在 small-sample / 接近边界时收敛得明显更好。

## 数据：什么是真的提升，什么是过度推销

35 题完整消融跑出来，r0 vs baseline：

| 指标 | baseline | pipeline-r0 | Cohen's d |
|---|---|---|---|
| factual_accuracy | 0.960 | 1.000 | **+0.60** |
| citation_coverage | 0.000 | 0.393 | **+8.65** |
| judge_overall | 4.240 | 4.299 | +0.13 |

`citation_coverage` 的 d=8.65（远超 0.8 的"巨大效应"阈值）是**真实可写**的简历数字，因为 baseline 直接是 0（单 prompt 没机会引用），pipeline 上来就是 39.3%。

`factual_accuracy +4 个点 / d=0.60` 也是中等效应，可写。

`judge_overall` 改善只有 d=0.13，**不能写**"显著提升"。

Red-Blue 对抗修复后的 r2 vs r0 数据等完整 35 题 rerun 跑完后会更新到 `docs/ablation.md`。诚实地讲，在弱基座 mimo + 简单 bench（35 题对 mimo 已经接近天花板）上，K=2 轮 Red-Blue 的边际收益本来就有限。

## 工程教训汇总

1. **代码 invariant 永远要兜住 LLM 的 prompt** — Red-Blue 翻车是最好的例子
2. **评测的合法性比数字大小更重要** — Judge 必须异构，端点不可用时降级要清楚标注
3. **小样本 + BCa bootstrap 比 percentile 准** — 不要为了简单选错
4. **Tavily 缓存要早加** — Planner 输出非确定，二跑很难命中相同 query；缓存只能省第二轮**完全相同**的 query
5. **状态机 + ResultEnvelope 是异步并发系统的两个保命神器** — 异常隔离 + 强制状态转移让调试快 3 倍
6. **Cohen's d 配 paired 计算才有意义** — 同一题在两个配置下打分，Δ 才能消除题目难度差异

## 仓库与复现

- GitHub: <https://github.com/your-handle/deepresearch-multiagent>
- 一键复现：`python experiments/run_ablation.py`
- 仅重跑 Red-Blue（不消耗 Tavily）：`python experiments/rerun_red_blue.py --src <prev_r0_reports>`
- 单元测试：`pytest tests/`（60+ 测试，覆盖状态机、池、压缩、对抗、统计等）

## 下一步

- 加 RAG 阶段引入更高质量来源（学术论文 PDF 而不是中文博客）
- 在更难的 bench（HotpotQA dev 子集）上验证压缩与 Red-Blue 的边际效用
- 多语言 bench：当前 ResearchBench 中英混合 35 题，扩到 100+ 时分领域 effect size 会更稳
- 探索"Multi-Critic"替代 Red-Blue：不同 persona 评审取交集而不是单一 Red

---

如果你也在做类似的 Agent 编排项目，欢迎 issue 讨论 prompt 调优 / 评测设计 / 异常隔离的具体取舍。
