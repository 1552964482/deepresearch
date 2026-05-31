# Requirements Document

## Introduction

DeepResearch-MultiAgent 是一个面向复杂深度研究任务的多智能体协作系统。给定一个研究问题，系统通过规划、检索、阅读压缩、写作、对抗评审的全链路 Agent 协作，输出一份结构化、带引用、经过自动化评测的研究报告。

项目目标是在弱基座（mimo-v2.5-pro，4 key × 16 并发）之上通过工程化 Agent 编排，达到接近或超越强基座（GPT-5 级）单点直答的报告质量，并产出可量化的评测证据。

最终交付物：可执行的 Python 项目（CLI + 可选 Streamlit demo）、ResearchBench 评测结果（含统计显著性）、消融实验报告、技术博客。

## Glossary

- **Agent**：一个具备特定职责（Planner / Searcher / Reader / Writer / Red / Blue / Reviewer）的异步可调用单元
- **DAG**：用于描述 Agent 之间数据依赖与并发关系的有向无环图
- **SubTask**：Planner 拆分出的子研究问题，每个 SubTask 拥有独立的 Searcher → Reader → Writer 子图
- **MimoPool**：自研的多 key 负载均衡 + 令牌桶限流的 mimo LLM 调用池
- **Judge**：独立于被测模型族的评测模型（gpt-5.4），用于 LLM-as-Judge 打分
- **Patch**：Blue Agent 对报告草稿的结构化修改单元（ADD / DELETE / MODIFY / VERIFY）
- **ResearchBench**：自建的研究报告评测集（11 领域 × 35 题）

## Requirements

### Requirement 1：多 Agent 编排引擎

**Objective**：作为系统使用者，我希望系统能够基于异步 DAG 调度多个 Agent 并发执行研究任务，并在子任务失败时自动降级，以便在不稳定的 LLM 后端下仍能稳定产出报告。

#### Acceptance Criteria

1. WHEN 用户提交一个研究 query，THEN 系统 SHALL 通过 Planner Agent 在 60 秒内输出 3-8 个 SubTask 的拆解
2. WHEN 多个 SubTask 进入执行阶段，THEN 系统 SHALL 通过 asyncio + Semaphore 控制全局 LLM 调用并发不超过 16
3. WHEN 某个 SubTask 单步执行超过配置的超时阈值（默认 90s），THEN 系统 SHALL 触发 L1 降级（兜底 LLM 直答）
4. WHEN 同一 query 下批量 SubTask 失败比例超过 30%，THEN 系统 SHALL 触发 L2 降级（重规划，缩减 SubTask 数量后重试一轮）
5. WHEN 整个 pipeline 运行超过全局上限（默认 10 分钟），THEN 系统 SHALL 触发 L3 强制收敛，输出当前最优草稿
6. WHEN 任意 Agent 节点抛出未捕获异常，THEN 系统 SHALL 通过 ResultEnvelope 隔离异常，不传播至其他并发分支
7. WHERE 任务进入新状态，THE 系统 SHALL 在 9 个状态（IDLE / PLANNING / SEARCHING / READING / COMPRESSING / WRITING / RED_REVIEW / BLUE_REVISE / DONE）以及异常态（FAILED / TIMEOUT）之间进行受控转移，并记录转移日志
8. WHEN 用户运行调试命令，THEN 系统 SHALL 能将状态机转移图导出为 mermaid 文本

### Requirement 2：Red-Blue 对抗降噪

**Objective**：作为研究报告质量的把关者，我希望系统能够通过双 Agent 对抗机制自动发现并修复报告中的事实、逻辑、引用问题，以便提升最终报告的可信度。

#### Acceptance Criteria

1. WHEN 报告草稿生成完成，THEN Red Agent SHALL 输出符合预定义 JSON Schema 的攻击列表，每条攻击包含 type（factual / logic / citation）、span（被攻击的文本片段）、evidence（攻击依据）、severity（0-1 浮点数）
2. WHEN Red Agent 返回的 JSON 不合法，THEN 系统 SHALL 依次尝试：原样重试一次 → 严格 JSON 模式重试一次 → 启发式正则提取 fallback，三层均失败才标记为攻击为空
3. WHEN Blue Agent 接收到攻击列表，THEN Blue Agent SHALL 对每条攻击输出一个 Patch（动作 ∈ {ADD, DELETE, MODIFY, VERIFY}）以及修改后的草稿增量
4. WHEN 一轮 Red-Blue 对抗完成，THEN 系统 SHALL 记录该轮攻击数、被接受的 patch 数、攻击命中类型分布以及 LLM-as-Judge 质量分变化
5. WHERE 配置的对抗轮数上限为 K（默认 K=2），THE 系统 SHALL 在第 K 轮结束后停止对抗循环，即使仍有剩余攻击
6. WHEN 某轮对抗后 Judge 质量分相对上一轮下降，THEN 系统 SHALL 回滚至上一轮草稿并停止对抗循环

### Requirement 3：三级上下文压缩与共享记忆

**Objective**：作为系统设计者，我希望在长链路 Agent 协作中通过分级语义压缩与跨 Agent 共享记忆，控制 prompt token 体积同时不丢失关键信息，以便系统能在长上下文任务下稳定运行。

#### Acceptance Criteria

1. WHEN Reader Agent 接收到从 Searcher 传入的网页文本块，THEN 系统 SHALL 通过 L1 Embedding 余弦相似度过滤掉与 SubTask query 相关性低于阈值（默认 0.45）的 chunk
2. WHEN L1 过滤后剩余 chunk 总 token 数仍超过预算，THEN 系统 SHALL 在每个 chunk 内执行 L2 TextRank 句子级抽取，保留 top-k（默认 top-8）关键句
3. WHERE chunk 中包含被识别为核心事实（数字 / 引用原句 / 命名实体）的句子，THE 系统 SHALL 在 L3 阶段保留原文不压缩
4. WHEN Agent 准备写入共享记忆，THEN 系统 SHALL 先查询 top-3 相似已写记录，IF 最高相似度 > 0.92 THEN 跳过写入并标记为 duplicate
5. WHEN 系统检测到两条记忆条目主语相同但谓语 embedding 余弦距离反向超过阈值，THEN 系统 SHALL 触发矛盾标记并将两条记录加入 review 队列
6. WHEN 系统执行向量检索，THEN 系统 SHALL 使用 SQLite + numpy 暴力 cosine（不引入 Milvus / Qdrant 等外部向量库），且单次检索（≤1 万条记录）耗时低于 100ms

### Requirement 4：完整评测体系

**Objective**：作为评估系统效果的研究者，我希望有一套自动化、可重复、有统计显著性的评测流水线，以便量化系统改进并支撑简历中的实验数字。

#### Acceptance Criteria

1. WHEN 评测被触发，THEN 系统 SHALL 加载自建 ResearchBench（11 领域 × 35 题）以及 HotpotQA 的 200 题子集
2. WHEN 评测某条样本，THEN 系统 SHALL 计算规则指标：事实准确率（关键事实匹配率）、幻觉率（无引用支撑的断言比例）、引用覆盖率（断言-引用对齐率）
3. WHEN 评测某条样本，THEN 系统 SHALL 调用 Judge（gpt-5.4，独立 client）按 5 维度（准确性 / 完整性 / 逻辑性 / 引用质量 / 可读性）打 1-5 分
4. WHERE 同一条样本的 LLM-as-Judge 评分，THE 系统 SHALL 跑 3 次取多数投票或均值（具体策略可配置），以降低 Judge 自身方差
5. WHEN 评测两组系统配置，THEN 系统 SHALL 通过 Bootstrap 重采样（默认 1000 次）输出 95% CI 以及 Cohen's d 效应量
6. WHEN 评测启动，THEN 系统 SHALL 支持 4 类后端热切换：mimo-v2.5-pro / 本地 vLLM 启动的 Qwen / DeepSeek API / OpenAI 兼容 API（gpt-5.4），通过配置文件切换不修改代码
7. WHEN 评测完成，THEN 系统 SHALL 输出 CSV 形式的逐题结果以及 Markdown 形式的汇总报告（含分领域得分、CI、效应量）

### Requirement 5：LLM 网关与多 key 负载均衡

**Objective**：作为系统运维者，我希望主链路调用通过自研 MimoPool 在 4 个 mimo key 之间负载均衡并实施令牌桶限流，以便最大化利用 100 RPM × 4 key 的配额而不触发限流。

#### Acceptance Criteria

1. WHEN 调用方发起一次 mimo 请求，THEN MimoPool SHALL 通过 least-in-flight 策略选择当前在飞请求最少的 key
2. WHERE 每个 key 拥有独立的 asyncio.Semaphore（默认 4）以及独立的滑动窗口令牌桶（100 RPM），THE 调用 SHALL 同时受这两个机制约束
3. WHEN 单个 key 触发限流（HTTP 429 或自定义错误），THEN MimoPool SHALL 在该 key 上指数退避，并将后续请求转移至其他可用 key
4. WHEN 一次请求失败（非限流类错误），THEN MimoPool SHALL 在同 key 上重试一次，IF 仍失败 THEN 切换至另一 key 重试一次，IF 三次均失败 THEN 抛出聚合异常
5. WHEN 系统运行期间，THEN MimoPool SHALL 在内存中维护每 key 的指标：累计请求数、成功率、平均延迟、当前在飞数、近 1 分钟 RPM 使用率
6. WHERE Judge 调用使用独立的 JudgeClient，THE Judge 流量 SHALL NOT 进入 MimoPool 队列，以避免被测流量挤占评测预算
7. WHEN 系统启动，THEN 系统 SHALL 从 .env 中读取 4 个 OPENAI_API_KEY* 变量，IF 缺失任一 key THEN 在 INFO 级别记录并以剩余 key 继续运行

### Requirement 6：CLI 与项目可运行性

**Objective**：作为项目使用者与简历展示者，我希望项目通过简单的 CLI 命令即可演示完整流程，以便面试演示和外部用户复现。

#### Acceptance Criteria

1. WHEN 用户运行 `dr-agent run "<query>"`，THEN 系统 SHALL 启动完整 pipeline 并将最终报告写入 `reports/<timestamp>-<slug>.md`
2. WHEN 用户运行 `dr-agent eval --bench researchbench`，THEN 系统 SHALL 启动评测流水线并将结果写入 `experiments/results/<timestamp>/`
3. WHEN 用户运行 `dr-agent state-graph`，THEN 系统 SHALL 输出当前状态机的 mermaid 文本到 stdout
4. WHEN 用户运行 `dr-agent pool-stats`，THEN 系统 SHALL 输出 MimoPool 各 key 的实时指标
5. WHERE 项目根目录存在 `.env`，THE 系统 SHALL 自动加载环境变量，无需用户手动 export
