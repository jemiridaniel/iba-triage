"""LangGraph triage pipeline.

    intake -> rules_pre -> retrieve -> outbreak -> reason -> rules_post -> compose

    deps = build_deps()
    result = run_triage(TriageRequest(text="...", state="Ondo"), deps)

Every node runs inside track_usage(), and its model, reasoning setting, tokens, latency and
cost become one TraceStep in result.decision_trace.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from backend.app.config import Settings, get_settings
from backend.app.graph import nodes
from backend.app.graph.state import Deps, TriageRequest, TriageState
from backend.app.llm.client import LLMClient, track_usage
from backend.app.llm.router import EvalConfig, route
from backend.app.rag.fake_embed import HASH_EMBED_MODEL, HashEmbedder
from backend.app.rag.store import VectorStore
from backend.app.schemas import TRIAGE_LABELS, DecisionTrace, TraceStep, TriageResult
from backend.app.tools.outbreak import FixtureSearch, OutbreakTool, SearchClient, TavilySearch

logger = logging.getLogger("iba.pipeline")

NodeFn = Callable[[TriageState, Deps], dict[str, Any]]

# step -> (node function, trace kind, router step for model/reasoning, if any)
STEPS: dict[str, tuple[NodeFn, str, str | None]] = {
    "intake": (nodes.intake, "llm", "intake"),
    "rules_pre": (nodes.rules_pre, "rules", None),
    "retrieve": (nodes.retrieve, "retrieval", None),
    "outbreak": (nodes.outbreak, "tool", "outbreak"),
    "reason": (nodes.reason, "llm", "reason"),
    "rules_post": (nodes.rules_post, "rules", None),
    "compose": (nodes.compose, "llm", "compose"),
}


def traced(step: str, fn: NodeFn, kind: str, router_step: str | None, deps: Deps):
    def node(state: TriageState) -> dict[str, Any]:
        start = time.perf_counter()
        with track_usage() as records:
            update = fn(state, deps)
        latency_ms = (time.perf_counter() - start) * 1000
        reasoning = None
        if router_step is not None:
            try:
                reasoning = route(router_step, deps.settings, state.eval_config).reasoning
            except Exception:  # model unset: the node already reported it
                reasoning = None
        costs = [r.cost_usd for r in records if r.cost_usd is not None]
        reasoning_counts = [r.reasoning_tokens for r in records if r.reasoning_tokens is not None]
        step_trace = TraceStep(
            step=step,
            kind=kind,  # type: ignore[arg-type]
            model=records[0].model if records else None,
            reasoning=reasoning if records else None,
            calls=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            reasoning_tokens=sum(reasoning_counts) if reasoning_counts else None,
            latency_ms=round(latency_ms, 1),
            cost_usd=round(sum(costs), 8) if costs else None,
            cached=bool(records) and all(r.cached for r in records),
            note=update.pop("_note", None) or _error_note(update),
        )
        logger.info(
            "step=%s model=%s calls=%d in=%d out=%d latency_ms=%.0f cost_usd=%s",
            step,
            step_trace.model,
            step_trace.calls,
            step_trace.prompt_tokens,
            step_trace.completion_tokens,
            step_trace.latency_ms,
            step_trace.cost_usd,
        )
        return {**update, "trace": [step_trace]}

    return node


def _error_note(update: dict[str, Any]) -> str | None:
    for key in ("intake_error", "reason_error", "compose_error", "retrieval_note"):
        if update.get(key):
            return str(update[key])[:300]
    return None


def build_graph(deps: Deps):
    graph = StateGraph(TriageState)
    for step, (fn, kind, router_step) in STEPS.items():
        graph.add_node(step, traced(step, fn, kind, router_step, deps))
    graph.set_entry_point("intake")
    graph.add_edge("intake", "rules_pre")
    graph.add_conditional_edges(
        "rules_pre",
        nodes.after_rules_pre,
        {"needs_info": END, "retrieve": "retrieve", "rules_post": "rules_post"},
    )
    graph.add_edge("retrieve", "outbreak")
    graph.add_edge("outbreak", "reason")
    graph.add_edge("reason", "rules_post")
    graph.add_edge("rules_post", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


def run_triage(
    request: TriageRequest, deps: Deps, eval_config: EvalConfig = "routed"
) -> TriageResult:
    final = build_graph(deps).invoke(TriageState(request=request, eval_config=eval_config))
    return to_result(TriageState.model_validate(final) if isinstance(final, dict) else final)


def to_result(state: TriageState) -> TriageResult:
    trace = DecisionTrace(steps=state.trace)
    if state.post is None:  # stopped for follow-up questions
        return TriageResult(
            status="needs_info",
            questions=state.questions,
            case=state.case,
            danger_signs=state.pre.danger_signs if state.pre else [],
            decision_trace=trace,
        )
    post = state.post
    warnings = list(post.warnings)
    if state.compose_error:
        warnings.append("Summary written from a template (the summary model failed).")
    return TriageResult(
        status=post.status,
        triage_level=post.triage_level,
        triage_label=TRIAGE_LABELS[post.triage_level],
        triage_rationale=post.triage_rationale,
        case=state.case,
        danger_signs=post.danger_signs,
        lassa_suspected=post.lassa_suspected,
        outbreak=state.outbreak,
        differential=post.differential,
        actions=post.actions,
        doses=post.doses,
        citations=post.citations,
        summary=state.compose.summary if state.compose else None,
        referral_note=state.compose.referral_note if state.compose else None,
        warnings=warnings,
        decision_trace=trace,
    )


# --- dependencies -----------------------------------------------------------


def build_deps(
    settings: Settings | None = None,
    *,
    index_dir: Path | None = None,
    mock_outbreak: Path | None = None,
    client: LLMClient | None = None,
    search: SearchClient | None = None,
) -> Deps:
    settings = settings or get_settings()
    client = client or LLMClient(settings)

    store, embed = None, None
    index_dir = index_dir or settings.index_dir
    if (index_dir / "meta.json").exists():
        store = VectorStore.load(index_dir)
        if store.model == HASH_EMBED_MODEL:
            embed = HashEmbedder()
        elif settings.model_embed:
            if store.model and store.model != settings.model_embed:
                logger.warning(
                    "index built with %s but MODEL_EMBED=%s", store.model, settings.model_embed
                )
            model = settings.model_embed
            embed = lambda texts: client.embed(texts, model=model, step="retrieve")  # noqa: E731

    if search is None:
        mock = mock_outbreak or settings.outbreak_mock_file
        if mock is not None:
            search = FixtureSearch(mock)
        elif settings.tavily_api_key is not None and settings.tavily_api_key.get_secret_value():
            search = TavilySearch(settings.tavily_api_key.get_secret_value())

    outbreak = OutbreakTool(client, search, settings.cache_dir)
    return Deps(settings=settings, client=client, outbreak=outbreak, store=store, embed=embed)
