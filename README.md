# DeepResearch-MultiAgent

面向复杂深度研究任务的多智能体协作系统。给定一个研究问题，系统通过 **规划 → 检索 → 阅读压缩 → 写作 → Red-Blue 对抗评审** 的全链路 Agent 协作，输出一份带引用的结构化研究报告，并通过自动化评测体系量化质量。

> **基线：** mimo-v2.5-pro（弱基座）；**评测器：** GPT-5.4（独立模型族，避免 self-preference bias）。
> **完整规格：** [`docs/spec/`](./docs/spec)

## 工程亮点

- **递归深挖（Recursive Deep-Dive）**：GapAnalyzer 分析每个 SubTask 的事实充分性，不足则生成追问子问题，按 depth/breadth 递归检索+阅读，全局搜索预算硬上限保护配额。把单层 RAG 升级为真正的 deep research（对标 GPT-Researcher 的 breadth/depth）。
- **MimoPool**：自研 4-key 负载均衡 + 每 key 滑窗令牌桶（100 RPM）+ 信号量并发控制（4/key），最少在飞调度，全局 16 并发
- **9 状态状态机**：表驱动转移，所有合法/非法迁移可静态枚举，可导出 mermaid 图
- **三级降级策略**：单 SubTask 超时 → L1 兜底 LLM 直答 / 批量失败 > 30% → L2 重规划 / 全局超时 → L3 强制收敛
- **Red-Blue 对抗降噪**：4 维度结构化 JSON 攻击（factual / logic / citation / completeness）+ 4 类 Patch 动作（ADD / DELETE / MODIFY / VERIFY）+ K 轮循环 + 引用保留 invariant
- **Multi-Critic 多角色评审**（Red 的进阶替代）：3 个 persona critic（事实核查 / 逻辑审稿 / 引用审计）并行攻击 + 双信号共识聚合（span 去重 + section 热点加权），主张精度优先；通过 `--reviewer multi` 启用
- **三层 JSON 解析 fallback**：直接 → strict-mode 重试 → 正则提取，对抗 LLM 输出格式抖动
- **三级语义压缩**：L1 Embedding 余弦过滤（bge-small-zh）→ L2 TextRank 句子级抽取 → L3 命名实体 / 数字 / 引用原句保留
- **跨 Agent 共享记忆**：SQLite + numpy 暴力 cosine（万级以下检索 < 100ms）+ 预写去重（cos > 0.92）+ 启发式矛盾检测
- **完整评测体系**：自建 ResearchBench（11 领域 × 35 题）+ 规则指标（事实准确率 / 幻觉率 / 引用覆盖率）+ LLM-as-Judge 5 维度 × n_samples 自一致性 + Bootstrap (BCa) 95% CI + Cohen's d
- **Judge 降级保护**：独立 Judge 端点不可用时自动降级到 mimo，并标注 `self_bias_risk=True`，保证评测合法性可追溯

## 实验结果（ResearchBench, n=35, paired bootstrap BCa）

| 配置 | factual_acc | hallu_rate | cite_cov | judge_overall |
|---|---|---|---|---|
| baseline (single-prompt) | 0.960 | 0.146 | 0.000 | 4.240 |
| pipeline-r0 (full DAG) | 1.000 | 0.164 | 0.393 | 4.299 |
| **pipeline-r2 (full DAG + Red-Blue K=2)** | **1.000** | **0.158** | **0.472** | **4.329** |

**关键效应**：

| 对比 | 指标 | Δ | Cohen's d |
|---|---|---|---|
| r0 vs baseline | citation_coverage | +0.393 | **+8.65 (巨大)** |
| r0 vs baseline | factual_accuracy | +0.040 | +0.60 (中等) |
| r2 vs r0 | citation_coverage | +0.079 | **+1.31 (巨大)** |
| r2 vs r0 | factual_accuracy | +0.000 | preserved |

详细消融与 Red-Blue 对抗效果分析见 [`docs/ablation.md`](./docs/ablation.md)。

## 快速开始

```bash
# 1. 装依赖
pip install -e .
pip install sentence-transformers tavily-python trafilatura pymupdf scikit-learn jieba sumy

# 2. 准备 .env（参考 .env.example）
#    - 4 个 mimo key （OPENAI_API_KEY1..4）
#    - JUDGE_API_KEY（独立 Judge 端点，建议 gpt-5/Claude 等异构模型族）
#    - TAVILY_API_KEY（搜索）

# 3. 跑一份研究报告
dr-agent run "什么是 GRPO 算法及其相对 PPO 的核心改进" --grounded --review 2

# 3b. 递归深挖（depth=2 两层追问，breadth=2 每层最多 2 个追问，预算 16 次搜索）
dr-agent run "什么是向量数据库 HNSW 索引" --depth 2 --breadth 2 --deepdive-budget 16

# 3c. Multi-Critic 多角色评审
dr-agent run "你的研究问题" --review 2 --reviewer multi

# 4. 跑评测（35 题）
dr-agent eval --bench researchbench --mode pipeline --review 2

# 5. 一键消融实验（baseline / r0 / r2 三组）
python experiments/run_ablation.py
```

## 项目结构

```
src/dr_agent/
├── cli.py                       # Typer CLI 入口
├── config.py                    # .env 加载、配置单例
├── llm/
│   ├── pool.py                  # MimoPool（4-key 负载均衡）
│   ├── judge.py                 # JudgeClient（独立模型族 + fallback）
│   ├── token_bucket.py          # 滑窗 RPM 限流
│   └── errors.py
├── orchestrator/
│   ├── state_machine.py         # 9 状态状态机
│   ├── envelope.py              # 异常隔离容器
│   ├── runner.py                # 完整 DAG 执行器
│   └── red_blue_loop.py         # K 轮 Red-Blue 对抗
├── agents/
│   ├── planner.py / searcher.py / reader.py / writer.py
│   ├── red.py                   # 4 维结构化攻击 + 三层 JSON fallback
│   └── blue.py                  # 4 类 patch + 引用保留 invariant
├── memory/
│   ├── embedder.py              # bge-small-zh + LRU 缓存
│   ├── store.py                 # SQLite + numpy 向量
│   └── compress.py              # L1/L2/L3 三级压缩
├── tools/
│   ├── search.py                # Tavily + 缓存
│   ├── search_cache.py          # SQLite Tavily 缓存
│   └── fetcher.py               # trafilatura + PyMuPDF + SSRF 防护
└── eval/
    ├── bench.py                 # ResearchBench 加载
    ├── rule_metrics.py          # 事实准确率 / 幻觉率 / 引用覆盖率
    ├── stats.py                 # Bootstrap + Cohen's d
    ├── compare.py               # 跨配置对比
    └── runner.py                # Eval 主流程

benchmarks/researchbench/
└── questions.jsonl              # 11 领域 × 35 题，含 reference_facts / forbidden_claims / scoring_rubric

experiments/
├── run_ablation.py              # 三配置消融驱动
├── rerun_red_blue.py            # 仅重跑 Red-Blue 模块（节省 Tavily 配额）
└── results/                     # 评测产出（CSV + summary md/json + reports）

tests/                            # 60+ 单元测试，覆盖状态机 / 池 / 压缩 / 统计 / 对抗 / 缓存
```

## 文档

- 需求与验收：[`docs/spec/requirements.md`](./docs/spec/requirements.md)
- 设计与组件接口：[`docs/spec/design.md`](./docs/spec/design.md)
- 实现任务清单：[`docs/spec/tasks.md`](./docs/spec/tasks.md)
- 详细配置与运行指南：[`docs/setup.md`](./docs/setup.md)
- 架构图：[`docs/architecture.md`](./docs/architecture.md)
- 消融与效应分析：[`docs/ablation.md`](./docs/ablation.md)
- 技术博客：[`docs/blog.md`](./docs/blog.md)

## License

MIT
