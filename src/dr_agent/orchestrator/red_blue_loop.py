"""K-round Red-Blue adversarial loop with optional in-loop quality rollback.

Workflow per round:
  1. Red attacks the current draft -> List[Attack]
  2. If 0 attacks -> stop early (DONE)
  3. Blue produces patches and applies them -> revised draft
  4. (optional) Judge scores both drafts; if quality dropped, rollback
     and stop.

Default behavior (no Judge calls) keeps the loop cheap and deterministic
during development. Pass ``judge=...`` and ``in_loop_judge=True`` to
enable the quality-rollback guard during evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from dr_agent.agents.base import AgentContext
from dr_agent.agents.blue import Blue, BlueInput
from dr_agent.agents.critics import MultiCritic, MultiCriticInput
from dr_agent.agents.red import Red, RedInput
from dr_agent.llm.judge import JudgeClient
from dr_agent.orchestrator.state_machine import StateMachine
from dr_agent.schemas.attack import Attack
from dr_agent.schemas.report import ResearchReport

if TYPE_CHECKING:
    from dr_agent.memory.embedder import Embedder


@dataclass
class RoundStat:
    round_idx: int
    n_attacks: int
    n_attacks_factual: int
    n_attacks_logic: int
    n_attacks_citation: int
    n_patches_accepted: int
    n_patches_skipped: int
    parse_strategy: str
    judge_score_overall: float | None = None  # only if in_loop_judge
    # Multi-critic extras (0 in single-Red mode)
    n_consensus: int = 0


@dataclass
class RedBlueResult:
    final_report: ResearchReport
    rounds: list[RoundStat] = field(default_factory=list)
    rolled_back: bool = False
    stopped_early: bool = False
    stop_reason: str = ""
    reviewer: str = "red"


async def _review_once(
    reviewer: str,
    report: ResearchReport,
    ctx: AgentContext,
    red: Red,
    multi: "MultiCritic | None",
) -> tuple[list[Attack], str, int]:
    """Produce attacks via the configured reviewer.

    Returns (attacks, parse_strategy, n_consensus). n_consensus is 0 for
    single-Red mode.
    """
    if reviewer == "multi" and multi is not None:
        env = await multi._safe_run(MultiCriticInput(report=report), ctx)
        if not env.ok or env.value is None:
            return [], "error", 0
        return env.value.attacks, env.value.parse_strategy_used, env.value.n_consensus
    env = await red._safe_run(RedInput(report=report), ctx)
    if not env.ok or env.value is None:
        return [], "error", 0
    return env.value.attacks, env.value.parse_strategy_used, 0


async def run_red_blue_loop(
    report: ResearchReport,
    ctx: AgentContext,
    *,
    sm: StateMachine,
    max_rounds: int = 2,
    judge: JudgeClient | None = None,
    in_loop_judge: bool = False,
    user_query: str | None = None,
    reviewer: str = "red",
    embedder: "Embedder | None" = None,
) -> RedBlueResult:
    """Run up to ``max_rounds`` of adversarial review.

    ``reviewer`` selects the attacker:
      * "red"   — single Red agent (4 dimensions in one prompt)
      * "multi" — Multi-Critic (3 persona critics + consensus merge);
                  requires ``embedder``.

    The state machine ``sm`` is expected to be in WRITING state on entry;
    we transition it through RED_REVIEW -> BLUE_REVISE -> ... -> DONE.
    """
    current = report
    last_report = report
    rounds: list[RoundStat] = []
    last_score: float | None = None

    # Move from WRITING to RED_REVIEW once.
    if sm.can("draft_ok"):
        sm.fire("draft_ok")  # WRITING -> RED_REVIEW

    red = Red()
    blue = Blue()
    multi: MultiCritic | None = None
    if reviewer == "multi":
        if embedder is None:
            raise ValueError("reviewer='multi' requires an embedder")
        multi = MultiCritic(embedder)

    for r in range(max_rounds):
        # ---- Review (Red or Multi-Critic) ----
        attacks, strategy, n_consensus = await _review_once(
            reviewer, current, ctx, red, multi
        )
        by_type = {"factual": 0, "logic": 0, "citation": 0, "completeness": 0}
        for a in attacks:
            by_type[a.type.value] = by_type.get(a.type.value, 0) + 1
        logger.info(
            "round {}/{}: [{}] {} attacks (factual={}, logic={}, citation={}, completeness={}, consensus={}, parse={})",
            r + 1,
            max_rounds,
            reviewer,
            len(attacks),
            by_type.get("factual", 0),
            by_type.get("logic", 0),
            by_type.get("citation", 0),
            by_type.get("completeness", 0),
            n_consensus,
            strategy,
        )

        if not attacks:
            rounds.append(
                RoundStat(
                    round_idx=r + 1,
                    n_attacks=0,
                    n_attacks_factual=0,
                    n_attacks_logic=0,
                    n_attacks_citation=0,
                    n_patches_accepted=0,
                    n_patches_skipped=0,
                    parse_strategy=strategy,
                    judge_score_overall=last_score,
                    n_consensus=n_consensus,
                )
            )
            # No attacks -> DONE via no_attacks edge
            if sm.can("no_attacks"):
                sm.fire("no_attacks")
            return RedBlueResult(
                final_report=current,
                rounds=rounds,
                stopped_early=True,
                stop_reason="no_attacks",
                reviewer=reviewer,
            )

        if sm.can("has_attacks"):
            sm.fire("has_attacks")  # RED_REVIEW -> BLUE_REVISE

        # ---- Blue ----
        blue_env = await blue._safe_run(
            BlueInput(report=current, attacks=attacks), ctx
        )
        if not blue_env.ok or blue_env.value is None:
            logger.error("blue round {} failed: {}", r + 1, blue_env.error)
            # Fail the round but keep current draft; stop loop.
            if sm.can("stop"):
                sm.fire("stop")
            return RedBlueResult(
                final_report=current,
                rounds=rounds,
                stop_reason=f"blue_failed:{blue_env.error}",
                reviewer=reviewer,
            )
        new_report = blue_env.value.revised_report
        accepted = blue_env.value.apply_stats.accepted
        skipped = (
            blue_env.value.apply_stats.skipped_no_match
            + blue_env.value.apply_stats.skipped_invalid
        )

        # ---- Optional in-loop Judge with rollback ----
        judge_overall: float | None = None
        if in_loop_judge and judge is not None and user_query:
            score = await judge.score(
                question=user_query, report=new_report.to_markdown()
            )
            judge_overall = score.overall
            logger.info(
                "round {} judge overall={:.2f} (prev={})",
                r + 1,
                judge_overall,
                f"{last_score:.2f}" if last_score is not None else "n/a",
            )
            if last_score is not None and judge_overall < last_score:
                # Rollback
                logger.warning(
                    "round {} quality dropped ({:.2f} < {:.2f}); rolling back",
                    r + 1,
                    judge_overall,
                    last_score,
                )
                rounds.append(
                    RoundStat(
                        round_idx=r + 1,
                        n_attacks=len(attacks),
                        n_attacks_factual=by_type["factual"],
                        n_attacks_logic=by_type["logic"],
                        n_attacks_citation=by_type["citation"],
                        n_patches_accepted=accepted,
                        n_patches_skipped=skipped,
                        parse_strategy=strategy,
                        judge_score_overall=judge_overall,
                        n_consensus=n_consensus,
                    )
                )
                if sm.can("stop"):
                    sm.fire("stop")
                return RedBlueResult(
                    final_report=last_report,  # rollback
                    rounds=rounds,
                    rolled_back=True,
                    stop_reason="quality_dropped",
                    reviewer=reviewer,
                )
            last_score = judge_overall

        rounds.append(
            RoundStat(
                round_idx=r + 1,
                n_attacks=len(attacks),
                n_attacks_factual=by_type.get("factual", 0),
                n_attacks_logic=by_type.get("logic", 0),
                n_attacks_citation=by_type.get("citation", 0),
                n_patches_accepted=accepted,
                n_patches_skipped=skipped,
                parse_strategy=strategy,
                judge_score_overall=judge_overall,
                n_consensus=n_consensus,
            )
        )

        last_report = current
        current = new_report

        # If not the last round, transition back into RED_REVIEW
        if r < max_rounds - 1:
            if sm.can("another_round"):
                sm.fire("another_round")  # BLUE_REVISE -> RED_REVIEW

    # Loop exhausted normally.
    if sm.can("stop"):
        sm.fire("stop")
    return RedBlueResult(
        final_report=current,
        rounds=rounds,
        stop_reason="max_rounds",
        reviewer=reviewer,
    )
