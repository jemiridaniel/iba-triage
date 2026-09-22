"""Per-step model routing and reasoning control.

    cfg = route("intake")                       # model, reasoning, max_tokens, extra_body
    case, records = call_json(client, "intake", messages, PatientCase)

Default ("routed") table, from SPEC §3 and the smoke-test measurements:

    step      model         reasoning
    intake    MODEL_FAST    off   structured extraction; reasoning adds ~5x tokens and latency
    outbreak  MODEL_FAST    off   extraction from search results
    reason    MODEL_REASON  on    the one step that needs deep reasoning
    compose   MODEL_MID     off   plain-language summary + referral note

Eval configurations swap the models (reason-only: MODEL_REASON everywhere; fast-only:
MODEL_FAST everywhere) and keep each step's reasoning setting, so they differ only in model.
Reasoning flags are per step in config (REASONING_INTAKE=true etc.).

How reasoning is switched off (measured on Token Factory, Nemotron 3 Nano and 3.5 Lightning):
chat_template_kwargs={"enable_thinking": false} works on both; a "/no_think" system prompt is
ignored by both. With reasoning on, Nano returns it in `message.reasoning`; Lightning strips
it and reports `reasoning_tokens`. Either way a reply cut off mid-reasoning dumps raw
reasoning into `content`, which the client rejects (LLMTruncatedError).
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from backend.app.config import Settings, get_settings
from backend.app.llm.client import CallRecord, LLMClient, LLMError

THINKING_OFF_KWARGS: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
# Kept for the smoke test's comparison run; Token Factory ignores it (see module docstring).
NO_THINK_DIRECTIVE = "/no_think"

Step = Literal["intake", "outbreak", "reason", "compose"]
EvalConfig = Literal["routed", "reason-only", "fast-only"]

_STEP_MODEL: dict[Step, str] = {
    "intake": "model_fast",
    "outbreak": "model_fast",
    "reason": "model_reason",
    "compose": "model_mid",
}
_CONFIG_MODEL: dict[EvalConfig, str | None] = {
    "routed": None,
    "reason-only": "model_reason",
    "fast-only": "model_fast",
}


@dataclass(frozen=True)
class StepConfig:
    step: Step
    model: str
    reasoning: bool
    max_tokens: int

    @property
    def extra_body(self) -> dict[str, Any] | None:
        # Reasoning on = send nothing (the models' default); off = the measured switch.
        return None if self.reasoning else THINKING_OFF_KWARGS


def route(
    step: Step, settings: Settings | None = None, config: EvalConfig = "routed"
) -> StepConfig:
    settings = settings or get_settings()
    if step not in _STEP_MODEL:
        raise ValueError(f"unknown step {step!r}")
    model_field = _CONFIG_MODEL[config] or _STEP_MODEL[step]
    model = getattr(settings, model_field)
    if not model:
        raise LLMError(f"{step}: {model_field.upper()} is not set in .env")
    reasoning = bool(getattr(settings, f"reasoning_{step}"))
    max_tokens = (
        settings.max_tokens_reasoning_on if reasoning else settings.max_tokens_reasoning_off
    )
    return StepConfig(step=step, model=model, reasoning=reasoning, max_tokens=max_tokens)


def call_json[T: BaseModel](
    client: LLMClient,
    step: Step,
    messages: list[dict[str, str]],
    schema: type[T],
    config: EvalConfig = "routed",
) -> tuple[T, list[CallRecord]]:
    """Route `step` and make the structured call. Records carry step, model and cost."""
    cfg = route(step, client.settings, config)
    return client.chat_json(
        messages,
        schema,
        model=cfg.model,
        step=step,
        max_tokens=cfg.max_tokens,
        extra_body=cfg.extra_body,
    )
