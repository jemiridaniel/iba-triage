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
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from backend.app.config import Settings, get_settings
from backend.app.graph import nodes
from backend.app.graph.state import Deps, TriageRequest, TriageState
from backend.app.llm.client import LLMClient, track_usage
from backend.app.llm.router import EvalConfig, route
from backend.app.llm.spend import SpendLimitError
from backend.app.rag.fake_embed import HASH_EMBED_MODEL, HashEmbedder
from backend.app.rag.store import VectorStore
from backend.app.rules.danger_signs import assess
from backend.app.schemas import (
    TRIAGE_LABELS,
    DecisionTrace,
    PatientCase,
    TraceStep,
    TriageLevel,
    TriageResult,
)
from backend.app.tools.endemicity import Endemicity
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
            model_latency_ms=round(sum(r.latency_ms for r in records), 1) if records else None,
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
        {"needs_info": END, "outbreak": "outbreak", "rules_post": "rules_post"},
    )
    # Outbreak before retrieval: live/baseline context shapes the retrieval queries.
    graph.add_edge("outbreak", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "rules_post")
    graph.add_edge("rules_post", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


def run_triage(
    request: TriageRequest, deps: Deps, eval_config: EvalConfig = "routed"
) -> TriageResult:
    final = build_graph(deps).invoke(TriageState(request=request, eval_config=eval_config))
    return to_result(TriageState.model_validate(final) if isinstance(final, dict) else final)


def failsafe_result(request: TriageRequest, exc: Exception) -> TriageResult:
    """An unexpected error still yields a safe answer: Refer now, with rule-detected signs."""
    case = PatientCase(state=request.state, lga=request.lga, raw_text=nodes.full_text(request))
    rules = assess(case, [])
    return TriageResult(
        status="incomplete",
        triage_level=TriageLevel.REFER_NOW,
        triage_label=TRIAGE_LABELS[TriageLevel.REFER_NOW],
        triage_rationale=nodes.INCOMPLETE_RATIONALE,
        danger_signs=rules.danger_signs,
        warnings=[f"Iba hit an unexpected error ({type(exc).__name__}) and failed safe."],
    )


# --- streaming ----------------------------------------------------------------


def _event_payload(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """What the UI needs from each node, as JSON-ready data."""

    def dump(value: Any) -> Any:
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    trace = update.get("trace") or []
    payload: dict[str, Any] = {"trace": dump(trace[0]) if trace else None}
    if node == "intake":
        payload |= {
            "case": dump(update.get("case")),
            "questions": [dump(q) for q in update.get("questions", [])],
            "error": update.get("intake_error"),
        }
    elif node == "rules_pre":
        pre = update["pre"]
        payload |= {
            "floor": pre.floor.value if pre.floor else None,
            "floor_label": TRIAGE_LABELS[pre.floor] if pre.floor else None,
            "danger_signs": [dump(h) for h in pre.danger_signs],
        }
    elif node == "retrieve":
        payload |= {
            "passages": len(update.get("hits", [])),
            "queries": [dump(q) for q in update.get("retrieval", [])],
            "note": update.get("retrieval_note"),
        }
    elif node == "outbreak":
        payload |= {"outbreak": dump(update["outbreak"])}
    elif node == "reason":
        payload |= {"ok": update.get("reason") is not None, "error": update.get("reason_error")}
    elif node == "rules_post":
        post = update["post"]
        payload |= dump(post) | {"triage_label": TRIAGE_LABELS[post.triage_level]}
    elif node == "compose":
        compose = update.get("compose")
        payload |= {
            "summary": compose.summary if compose else None,
            "referral_note": compose.referral_note if compose else None,
            "error": update.get("compose_error"),
        }
    return payload


def stream_triage(
    request: TriageRequest, deps: Deps, eval_config: EvalConfig = "routed"
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) after every node, then ("final", TriageResult).

    Errors are events too: a SpendLimitError yields a fatal "error" and no triage result;
    any other unexpected exception yields "error" then a fail-safe "final" (Refer now).
    """
    values: dict[str, Any] | None = None
    try:
        stream = build_graph(deps).stream(
            TriageState(request=request, eval_config=eval_config),
            stream_mode=["updates", "values"],
        )
        for mode, chunk in stream:
            if mode == "values":
                values = chunk
                continue
            for node, update in chunk.items():
                yield node, _event_payload(node, update)
        state = TriageState.model_validate(values)
        yield "final", to_result(state).model_dump(mode="json")
    except SpendLimitError:
        logger.error("spend limit reached during stream")
        yield (
            "error",
            {"fatal": True, "message": "Iba is temporarily unavailable (spending limit reached)."},
        )
    except Exception as exc:
        logger.exception("pipeline error during stream")
        yield (
            "error",
            {"fatal": False, "message": f"Unexpected error ({type(exc).__name__}); failing safe."},
        )
        yield "final", failsafe_result(request, exc).model_dump(mode="json")


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
        grounding=post.grounding,
        retrieval=state.retrieval,
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
            instruction = settings.embed_query_instruction

            def embed(texts: list[str]) -> list[list[float]]:
                if instruction:
                    texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
                return client.embed(texts, model=model, step="retrieve")

    if search is None:
        mock = mock_outbreak or settings.outbreak_mock_file
        if mock is not None:
            search = FixtureSearch(mock)
        elif settings.tavily_api_key is not None and settings.tavily_api_key.get_secret_value():
            search = TavilySearch(settings.tavily_api_key.get_secret_value())

    def resolve(doc_id: str, phrase: str) -> str | None:
        chunk = store.find(doc_id, phrase) if store is not None else None
        return chunk.id if chunk else None

    outbreak = OutbreakTool(
        client,
        search,
        settings.cache_dir,
        endemicity=Endemicity.load(settings.endemicity_file),
        resolve=resolve,
    )
    return Deps(settings=settings, client=client, outbreak=outbreak, store=store, embed=embed)
