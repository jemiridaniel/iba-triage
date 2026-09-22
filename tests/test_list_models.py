"""Model catalogue parsing and .env.example price block. Uses a saved catalogue sample."""

import json
from pathlib import Path

import pytest

from scripts.list_models import (
    BLOCK_END,
    BLOCK_START,
    parse_models,
    prices_json,
    render_block,
    update_env_example,
)

FIXTURE = Path(__file__).parent / "fixtures" / "models_verbose_sample.json"


@pytest.fixture
def models():
    return parse_models(json.loads(FIXTURE.read_text()))


def by_id(models, model_id):
    return next(m for m in models if m.id == model_id)


def test_prices_converted_to_per_million(models) -> None:
    nano = by_id(models, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    assert nano.input_per_m == 0.06
    assert nano.output_per_m == 0.24
    assert nano.context_length == 262144


def test_types(models) -> None:
    assert by_id(models, "Qwen/Qwen3-Embedding-8B").type == "embedding"
    assert by_id(models, "nvidia/Nemotron-3-Ultra-550b-a55b").type == "chat"
    assert by_id(models, "openbmb/MiniCPM-V-4_5").type == "chat"  # text+image->text


def test_nvidia_models_sort_first(models) -> None:
    assert [m.is_nvidia for m in models[:4]] == [True] * 4
    assert models[4].is_embedding


def test_prices_json_covers_nvidia_and_embedding_only(models) -> None:
    prices = prices_json(models)
    assert set(prices) == {
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-3-Ultra-550b-a55b",
        "nvidia/Nemotron-3_5-Lightning",
        "nvidia/nemotron-3-super-120b-a12b",
        "Qwen/Qwen3-Embedding-8B",
    }
    assert prices["Qwen/Qwen3-Embedding-8B"] == {"input": 0.01, "output": 0.0}


def test_block_is_commented_and_loadable(models) -> None:
    block = render_block(prices_json(models))
    lines = block.splitlines()
    assert all(line.startswith("#") for line in lines)
    (price_line,) = [line for line in lines if line.startswith("# MODEL_PRICES=")]
    assert json.loads(price_line.removeprefix("# MODEL_PRICES="))


def test_update_inserts_after_model_prices_then_replaces_in_place() -> None:
    env = "A=1\nMODEL_PRICES={}\nB=2\n"
    once = update_env_example(env, render_block({"m": {"input": 1.0, "output": 2.0}}))
    assert once.index("MODEL_PRICES={}") < once.index(BLOCK_START) < once.index("B=2")
    twice = update_env_example(once, render_block({"n": {"input": 3.0, "output": 4.0}}))
    assert twice.count(BLOCK_START) == 1
    assert twice.count(BLOCK_END) == 1
    assert '"n"' in twice
    assert '"m"' not in twice
    assert twice.endswith("B=2\n")
