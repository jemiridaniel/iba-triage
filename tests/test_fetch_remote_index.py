"""backend/app/rag/fetch_remote_index: no-ops locally/on Nebius, fetches on Render. No network."""

from pathlib import Path

import pytest

from backend.app.rag import fetch_remote_index as mod


def test_noop_when_hf_index_repo_unset(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv("HF_INDEX_REPO", raising=False)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    assert mod.main() == 0
    assert "skipping" in capsys.readouterr().out


def test_noop_when_index_already_present(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "meta.json").write_text("{}")
    monkeypatch.setenv("HF_INDEX_REPO", "someone/some-index")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    assert mod.main() == 0
    assert "already present" in capsys.readouterr().out


def test_download_failure_is_a_loud_error_not_a_silent_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_INDEX_REPO", "someone/some-index")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    assert mod.main() == 1


def test_missing_meta_json_after_download_is_an_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_INDEX_REPO", "someone/some-index")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: None)
    assert mod.main() == 1


def test_successful_download(monkeypatch, tmp_path: Path) -> None:
    def fake_download(*, repo_id, repo_type, token, local_dir):
        assert repo_id == "someone/some-index"
        assert repo_type == "dataset"
        (Path(local_dir) / "meta.json").write_text("{}")

    monkeypatch.setenv("HF_INDEX_REPO", "someone/some-index")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert mod.main() == 0
    assert (tmp_path / "meta.json").exists()


def test_missing_dependency_is_a_clear_error(monkeypatch, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("no module")
        return real_import(name, *a, **kw)

    monkeypatch.setenv("HF_INDEX_REPO", "someone/some-index")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(builtins, "__import__", blocked)
    assert mod.main() == 1


if __name__ == "__main__":
    pytest.main([__file__])
