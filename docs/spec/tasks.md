# Implementation Plan

## Phase 1: 项目骨架与 LLM 网关（第 1 周）

- [ ] 1. 初始化项目结构与依赖
- [ ] 1.1 创建 `pyproject.toml`（Python ≥ 3.11，依赖：httpx、pydantic v2、loguru、typer、python-dotenv、numpy、scipy、tenacity、pytest、pytest-asyncio、scikit-learn、networkx、trafilatura、tavily-python、PyMuPDF、tiktoken、jieba、sumy、streamlit）
  - _Requirements: 6.1, 6.5_
- [ ] 1.2 在 `src/dr_agent/` 下创建完整目录骨架并写空 `__init__.py`
  - _Requirements: 6.1_
- [ ] 1.3 实现 `dr_agent/config.py`：从 `.env` 加载 4 个 mimo key、Judge 配置、各超时与并发参数；提供单例 `get_settings()`
  - _Requirements: 5.7, 6.5_
- [ ] 1.4 实现 `dr_agent/utils/logging.py`：loguru 配置 + key mask filter；`utils/trace.py`：trace_id 生成与 contextvar 传播
  - _Requirements: 1.7_

- [ ] 2. 实现 LLM 网关层
- [ ] 2.1 实现 `dr_agent/llm/token_bucket.py`：滑动窗口令牌桶（按时间戳队列），暴露 `acquire(timeout)` 异步接口
  - _Requirements: 5.2_
- [ ] 2.2 实现 `dr_agent/llm/retry.py`：基于 tenacity 的同 key 重试 + 切 key 重试装饰器，区分限流（429）与其他错误
  - _Requirements: 5.3, 5.4_
- [ ] 2.3 实现 `dr_agent/llm/pool.py` 的 `MimoPool`：4 key 各自 Semaphore(4) + 令牌桶(100/min)；`least-in-flight` 选 key；`chat()` 异步接口；指标收集（in-flight / 成功率 / 平均延迟 / RPM 用量）
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
- [ ] 2.4 实现 `dr_agent/llm/judge.py` 的 `JudgeClient`：独立 base_url + 独立 Semaphore；`score(question, report, rubric, n_samples=3)` 接口，自带方差降噪
  - _Requirements: 4.3, 4.4, 5.6_
- [ ] 2.5 编写 `tests/test_pool.py`：mock httpx，断言 least-in-flight 选 key 顺序、令牌桶限流、429 退避、3 次重试切 key
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

## Phase 2: Schemas、状态机与最小 DAG

- [ ] 3. 定义核心数据模型
- [ ] 3.1 实现 `dr_agent/schemas/task.py`：`SubTask`、`ResearchTask`、`AgentState`（Enum：IDLE/PLANNING/SEARCHING/READING/COMPRESSING/WRITING/RED_REVIEW/BLUE_REVISE/DONE/FAILED/TIMEOUT）
  - _Requirements: 1.7_
- [ ] 3.2 实现 `dr_agent/schemas/attack.py`：`AttackType`、`Attack` 模型（含 severity 0-1 校验）
  - _Requirements: 2.1_
- [ ] 3.3 实现 `dr_agent/schemas/patch.py`：`PatchAction` Enum、`Patch` 模型（按 action 字段做条件必填校验）
  - _Requirements: 2.3_
- [ ] 3.4 实现 `dr_agent/schemas/report.py`：`Citation`、`Section`、`ResearchReport`（带 `to_markdown()` 方法）
  - _Requirements: 6.1_

- [ ] 4. 实现 Orchestrator 核心
- [ ] 4.1 实现 `dr_agent/orchestrator/envelope.py`：`ResultEnvelope[T]` 泛型容器，封装 ok/err/value/elapsed/trace_id
  - _Requirements: 1.6_
- [ ] 4.2 实现 `dr_agent/orchestrator/state_machine.py`：表驱动 9 状态状态机，合法转移表 + 转移日志钩子；`export_mermaid()` 导出 mermaid 文本
  - _Requirements: 1.7, 1.8_
- [ ] 4.3 实现 `dr_agent/orchestrator/dag.py`：基于 networkx + asyncio 的轻量 DAG 调度器；支持节点超时、节点失败隔离；遵循全局 Semaphore 上限
  - _Requirements: 1.2, 1.6_
- [ ] 4.4 实现 `dr_agent/orchestrator/degrade.py`：L1（单步超时直答）、L2（批量失败重规划）、L3（全局超时强制收敛）三种降级策略实现
  - _Requirements: 1.3, 1.4, 1.5_
- [ ] 4.5 编写 `tests/test_state_machine.py`：表驱动测试所有合法 / 非法转移
  - _Requirements: 1.7_

- [ ] 5. 实现 Agent 基类
- [ ] 5.1 实现 `dr_agent/agents/base.py`：`AbstractAgent[InT, OutT]` 泛型基类、`AgentContext`（持有 pool / memory / trace_id）、prompt 渲染辅助
  - _Requirements: 1.1_

## Phase 3: 端到端最小 demo（hello world）

- [ ] 6. 实现最简 Planner 与 Writer（先不接检索）
- [ ] 6.1 实现 `dr_agent/agents/planner.py`：调用 mimo，输出 3-8 个 SubTask（带 JSON Schema 校验，失败回退为单 SubTask）
  - _Requirements: 1.1, 2.2_
- [ ] 6.2 实现 `dr_agent/agents/writer.py`：基于 SubTask 列表（无外部资料）撰写一份初版报告（markdown），输出 `ResearchReport`
  - _Requirements: 6.1_
- [ ] 6.3 实现 `dr_agent/cli.py`：`dr-agent run "<query>"` 命令；加载 `.env`、初始化 MimoPool、跑 Planner → Writer 最小 DAG、保存到 `reports/<ts>-<slug>.md`
  - _Requirements: 6.1, 6.5_
- [ ] 6.4 跑通 hello world：执行 `dr-agent run "什么是 GRPO 算法"`，确认产出一份合格 markdown 报告
  - _Requirements: 6.1_

## Phase 4: 检索 + 阅读 + 三级压缩 + 共享记忆

- [ ] 7. 实现工具层
- [ ] 7.1 实现 `dr_agent/tools/search.py`：Tavily 主搜索 + SerpAPI fallback（接口存在但未配 key 时优雅降级到本地 mock）；输出 `SearchResult` 列表
  - _Requirements: 1.1_
- [ ] 7.2 实现 `dr_agent/tools/fetcher.py`：trafilatura 抓网页正文 + PyMuPDF 抓 PDF；URL SSRF 白名单校验
  - _Requirements: 1.1_

- [ ] 8. 实现三级压缩
- [ ] 8.1 实现 `dr_agent/memory/compress.py` 的 L1 Embedding 余弦过滤（阈值默认 0.45）
  - _Requirements: 3.1_
- [ ] 8.2 实现 L2 TextRank 句子级抽取（中英文混排，使用 sumy + jieba 分词）
  - _Requirements: 3.2_
- [ ] 8.3 实现 L3 命名实体 / 数字 / 引用原句保留逻辑（基于正则 + spaCy 可选）
  - _Requirements: 3.3_
- [ ] 8.4 编写 `tests/test_compress.py`：构造已知 chunk，验证三级阈值与保留规则
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 9. 实现共享记忆存储
- [ ] 9.1 实现 `dr_agent/memory/store.py`：SQLite WAL 模式、表结构（id/task_id/agent_id/text/embedding_blob/created_at/is_duplicate/contradicts_with）；numpy 暴力 cosine 检索
  - _Requirements: 3.6_
- [ ] 9.2 实现 `dr_agent/memory/dedupe.py`：写入前 top-3 相似度查询，> 0.92 跳过
  - _Requirements: 3.4_
- [ ] 9.3 实现 `dr_agent/memory/contradict.py`：主语相同 + 谓语 embedding 反向超阈值时标记矛盾
  - _Requirements: 3.5_
- [ ] 9.4 编写 `tests/test_dedupe.py`、`tests/test_memory_store.py`：万级条目下检索 < 100ms 性能基准
  - _Requirements: 3.6_

- [ ] 10. 实现 Searcher 与 Reader
- [ ] 10.1 实现 `dr_agent/agents/searcher.py`：对每个 SubTask 调用 search.py + fetcher.py，产出 chunk 列表写入 memory
  - _Requirements: 1.1_
- [ ] 10.2 实现 `dr_agent/agents/reader.py`：从 memory 拉取 chunk → 调用 Compressor 三级压缩 → 输出 KeyFacts
  - _Requirements: 3.1, 3.2, 3.3_

## Phase 5: Red-Blue 对抗

- [ ] 11. 实现 Red / Blue Agent 与 K 轮循环
- [ ] 11.1 实现 `dr_agent/agents/red.py`：prompt 模板要求结构化 JSON 输出；三层 JSON 解析 fallback（原样 → strict JSON 模式 → 正则提取）
  - _Requirements: 2.1, 2.2_
- [ ] 11.2 实现 `dr_agent/agents/blue.py`：按 attack 列表逐条产出 Patch（ADD/DELETE/MODIFY/VERIFY），返回 patched draft
  - _Requirements: 2.3_
- [ ] 11.3 在 Orchestrator 中实现 K 轮对抗循环：每轮跑 Judge 打分，若分数下降则回滚至上一轮草稿并停止
  - _Requirements: 2.4, 2.5, 2.6_
- [ ] 11.4 编写 `tests/test_red_blue.py`：mock LLM 返回构造好的 attack/patch，验证 K 轮循环、回滚、JSON fallback 三层
  - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6_

## Phase 6: 评测体系

- [ ] 12. 构建 ResearchBench
- [ ] 12.1 设计 `benchmarks/researchbench/questions.jsonl`：11 领域 × 3-4 题，每条含 question / domain / reference_facts / scoring_rubric
  - _Requirements: 4.1_
- [ ] 12.2 写一个 `scripts/build_hotpotqa_subset.py`：从 HotpotQA dev 集中抽取 200 题写入 `benchmarks/hotpotqa/subset.jsonl`
  - _Requirements: 4.1_

- [ ] 13. 实现评测指标
- [ ] 13.1 实现 `dr_agent/eval/rule_metrics.py`：事实准确率（关键事实命中率）、幻觉率（无引用的断言比例，断言切分用 spaCy 句切）、引用覆盖率（断言-引用对齐率）
  - _Requirements: 4.2_
- [ ] 13.2 实现 `dr_agent/eval/judge_metrics.py`：5 维度 rubric prompt + n_samples 重采样取均值；返回 `JudgeScore` 含原始样本
  - _Requirements: 4.3, 4.4_
- [ ] 13.3 实现 `dr_agent/eval/stats.py`：Bootstrap 1000 次重采样计算 95% CI（BCa 修正）、Cohen's d 效应量
  - _Requirements: 4.5_
- [ ] 13.4 编写 `tests/test_stats.py`：与 scipy 标准实现对比验证

- [ ] 14. 实现评测主流程与多后端切换
- [ ] 14.1 实现 `dr_agent/eval/runner.py`：加载 bench → 对每条样本跑被测系统 → 计算规则指标 + Judge → Bootstrap 聚合 → 输出 CSV + Markdown
  - _Requirements: 4.5, 4.7_
- [ ] 14.2 实现后端配置切换：通过 `--backend mimo|qwen-vllm|deepseek|openai` CLI 参数动态选 backend；vLLM 后端需提供本地启动文档
  - _Requirements: 4.6_
- [ ] 14.3 在 `cli.py` 中加入 `dr-agent eval --bench researchbench --backend mimo` 命令
  - _Requirements: 6.2_

## Phase 7: 消融实验、UI 与对外发布

- [ ] 15. 消融实验脚本
- [ ] 15.1 编写 `experiments/ablation_red_blue.py`：对比"无 Red-Blue"vs"K=1"vs"K=2"vs"K=3"的报告质量分
  - _Requirements: 2.4, 4.5_
- [ ] 15.2 编写 `experiments/ablation_compression.py`：对比"无压缩"vs"L1 only"vs"L1+L2"vs"L1+L2+L3"的 token 占用 / 信息保留 / 报告质量
  - _Requirements: 3.1, 3.2, 3.3_
- [ ] 15.3 编写 `experiments/baseline_vs_pipeline.py`：mimo 单 prompt 直答 vs gpt-5.4 单 prompt 直答 vs mimo + 全 pipeline，三组对比
  - _Requirements: 4.5_

- [ ] 16. Streamlit Demo UI
- [ ] 16.1 实现 `ui/app.py`：左栏输入 query 与 backend 选择，右栏实时展示 SubTask 进度、状态机状态、最终报告 markdown 预览
  - _Requirements: 6.1_

- [ ] 17. 文档与发布
- [ ] 17.1 撰写 `README.md`：项目介绍、架构图、快速开始、评测结果、消融数据
  - _Requirements: 6.1_
- [ ] 17.2 撰写 `docs/architecture.md` 详细架构文档（含 mermaid 图）
  - _Requirements: 1.8_
- [ ] 17.3 撰写技术博客 `docs/blog-draft.md`（约 3000 字，重点讲 MimoPool 设计、Red-Blue 对抗效果、评测体系合法性）
  - _Requirements: 4.3, 4.4, 5.1_
- [ ] 17.4 完善 `.gitignore`（排除 `.env`、`reports/`、`experiments/results/`、`__pycache__`、`*.db`）
  - _Requirements: 6.5_
