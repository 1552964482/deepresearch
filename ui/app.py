"""Streamlit demo for DeepResearch-MultiAgent.

Run with:
    streamlit run ui/app.py

Two views:
  * **Live Run** — submit a research query and watch the pipeline produce
    a report; shows state transitions, Red-Blue rounds, and pool stats.
  * **Eval Browser** — load any ``experiments/results/*-ablation`` folder
    and render the summary tables and the three pairwise Cohen's d
    comparisons.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# OpenMP workaround must precede torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from dr_agent.config import get_settings
from dr_agent.llm.judge import JudgeClient
from dr_agent.llm.pool import MimoPool
from dr_agent.memory.embedder import Embedder
from dr_agent.memory.store import MemoryStore
from dr_agent.orchestrator.runner import run_grounded
from dr_agent.tools.fetcher import Fetcher
from dr_agent.tools.search import WebSearcher
from dr_agent.tools.search_cache import SearchCache
from dr_agent.utils.logging import setup_logging


st.set_page_config(
    page_title="DeepResearch-MultiAgent",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading bge-small-zh embedder…")
def _get_embedder() -> Embedder:
    e = Embedder()
    e.warmup()
    return e


def _live_view() -> None:
    setup_logging()
    settings = get_settings()
    st.title("DeepResearch-MultiAgent")
    st.caption(
        f"Backend: **{settings.mimo.model}** × {len(settings.mimo.api_keys)} keys "
        f"(total concurrency {settings.mimo.total_concurrency}) "
        f"| Judge: **{settings.judge.model}**"
    )

    with st.sidebar:
        st.subheader("Settings")
        review = st.slider("Red-Blue rounds (K)", 0, 4, 2)
        max_results = st.slider("Tavily results / SubTask", 3, 8, 5)
        run_btn = st.button("Run", type="primary")

    query = st.text_area(
        "Research question",
        value="什么是 GRPO 算法及其相对 PPO 的核心改进？",
        height=80,
    )

    if not run_btn:
        st.info("Enter a question on the left and click **Run** to start the pipeline.")
        return

    log_box = st.empty()
    progress = st.progress(0.0, text="Initializing…")

    async def _go() -> None:
        embedder = _get_embedder()
        cache = SearchCache(Path(".cache/tavily.db"))
        memory_path = Path(".cache/ui_memory.db")
        async with (
            MimoPool(settings.mimo) as pool,
            Fetcher() as fetcher,
        ):
            memory = MemoryStore(memory_path, embedder)
            try:
                searcher = WebSearcher(cache=cache)
                progress.progress(0.10, text="Planner running…")
                report, sm, rb_result = await run_grounded(
                    query,
                    pool,
                    embedder=embedder,
                    memory=memory,
                    web_searcher=searcher,
                    fetcher=fetcher,
                    config=settings.orch,
                    max_results_per_query=max_results,
                    review_rounds=review,
                )
            finally:
                memory.close()

        progress.progress(1.0, text="Done")
        st.success(f"Report generated: {len(report.sections)} sections, "
                   f"{len(report.citations)} citations")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Report")
            st.markdown(report.to_markdown())
        with c2:
            st.subheader("State trace")
            states = [sm.history[0].from_state.value] + [
                r.to_state.value for r in sm.history
            ]
            st.code(" → ".join(states))

            if rb_result and rb_result.rounds:
                st.subheader("Red-Blue rounds")
                for rs in rb_result.rounds:
                    st.write(
                        f"R{rs.round_idx}: attacks={rs.n_attacks} "
                        f"(F={rs.n_attacks_factual}, L={rs.n_attacks_logic}, "
                        f"C={rs.n_attacks_citation}); "
                        f"patches accepted={rs.n_patches_accepted}; "
                        f"parse={rs.parse_strategy}"
                    )
                if rb_result.rolled_back:
                    st.warning("Rolled back due to quality drop.")

            st.subheader("MimoPool stats")
            st.json(pool.stats())
            st.subheader("Tavily cache stats")
            st.json(cache.stats())
            cache.close()

    try:
        asyncio.run(_go())
    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {e}")


def _eval_view() -> None:
    st.title("Eval Browser")
    root = Path("experiments/results")
    if not root.exists():
        st.info("No eval runs found. Run `python experiments/run_ablation.py` first.")
        return

    runs = sorted([p for p in root.glob("*-ablation") if p.is_dir()], reverse=True)
    if not runs:
        st.info("No `*-ablation/` folder under experiments/results yet.")
        return

    pick = st.selectbox(
        "Pick an ablation run",
        runs,
        format_func=lambda p: p.name,
    )

    summary_path = pick / "ablation-summary.md"
    if summary_path.exists():
        st.markdown(summary_path.read_text(encoding="utf-8"))

    st.subheader("Pairwise comparisons")
    for cmp in sorted(pick.glob("compare-*.md")):
        with st.expander(cmp.stem):
            st.markdown(cmp.read_text(encoding="utf-8"))

    st.subheader("Per-config summaries")
    for cfg_name in ("baseline", "pipeline-r0", "pipeline-r2"):
        sjson = pick / cfg_name / "summary.json"
        if not sjson.exists():
            continue
        with st.expander(cfg_name):
            data = json.loads(sjson.read_text(encoding="utf-8"))
            st.json(data)


def main() -> None:
    page = st.sidebar.radio("View", ["Live Run", "Eval Browser"])
    if page == "Live Run":
        _live_view()
    else:
        _eval_view()


if __name__ == "__main__":
    main()
