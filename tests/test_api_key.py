"""xAI key precedence. Uses fake key strings only."""

import os
import subprocess
from pathlib import Path

from assistant import load_repo_env, resolve_api_key

ROOT = Path(__file__).resolve().parents[1]
FAKE_PROCESS = "fake-process-key"
FAKE_DOTENV = "fake-dotenv-key"
FAKE_SECRETS = "fake-secrets-key"


def _write_sources(directory: Path, dotenv: str | None, secrets: str | None) -> tuple[Path, Path]:
    env_path = directory / ".env"
    secrets_path = directory / "secrets.toml"
    if dotenv is not None:
        env_path.write_text(dotenv, encoding="utf-8")
    if secrets is not None:
        secrets_path.write_text(secrets, encoding="utf-8")
    return env_path, secrets_path


def test_process_env_wins_over_dotenv_and_secrets(tmp_path):
    env_path, secrets_path = _write_sources(
        tmp_path,
        "XAI_API_KEY=fake-dotenv-key\n",
        'xai_api_key = "fake-secrets-key"\n',
    )
    found = resolve_api_key(
        {"XAI_API_KEY": FAKE_PROCESS},
        env_path=env_path,
        secrets_path=secrets_path,
    )
    assert found == FAKE_PROCESS


def test_dotenv_wins_over_secrets_without_an_export(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    env_path, secrets_path = _write_sources(
        tmp_path,
        'export XAI_API_KEY="fake-dotenv-key"\n',
        "xai_api_key = 'fake-secrets-key'\n",
    )
    found = resolve_api_key({}, env_path=env_path, secrets_path=secrets_path)
    assert found == FAKE_DOTENV
    assert os.environ.get("XAI_API_KEY", "") == ""


def test_secrets_toml_is_read_without_streamlit(tmp_path):
    env_path, secrets_path = _write_sources(
        tmp_path,
        "OTHER=1\n# XAI_API_KEY is absent\n",
        'xai_api_key = "fake-secrets-key"\n',
    )
    found = resolve_api_key({}, env_path=env_path, secrets_path=secrets_path)
    assert found == FAKE_SECRETS


def test_blank_process_value_falls_through_to_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    env_path, secrets_path = _write_sources(
        tmp_path,
        "XAI_API_KEY=fake-dotenv-key # not a real key\n",
        'xai_api_key = "fake-secrets-key"\n',
    )
    found = resolve_api_key({"XAI_API_KEY": "   "}, env_path=env_path, secrets_path=secrets_path)
    assert found == FAKE_DOTENV
    assert "XAI_API_KEY" not in os.environ


def test_load_repo_env_fills_process_env_and_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_PROCESS)
    (tmp_path / ".env").write_text("XAI_API_KEY=fake-dotenv-key\n", encoding="utf-8")
    load_repo_env(tmp_path)
    assert os.environ["XAI_API_KEY"] == FAKE_PROCESS

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    load_repo_env(tmp_path)
    assert os.environ["XAI_API_KEY"] == FAKE_DOTENV
    assert resolve_api_key(secrets_path=tmp_path / "missing.toml") == FAKE_DOTENV


def test_streamlit_secret_is_used_when_files_are_absent(tmp_path, monkeypatch):
    import sys
    import types

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    fake = types.ModuleType("streamlit")
    fake.secrets = {"xai_api_key": FAKE_SECRETS}
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    found = resolve_api_key(env_path=tmp_path / "missing.env")
    assert found == FAKE_SECRETS


def test_missing_sources_return_none(tmp_path):
    assert (
        resolve_api_key(
            {},
            env_path=tmp_path / "missing.env",
            secrets_path=tmp_path / "missing.toml",
        )
        is None
    )


def test_gitignore_covers_env_and_secrets_not_example():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert ".env" in lines
    assert ".streamlit/secrets.toml" in lines
    assert ".env.example" not in lines
    assert not any(line in {".env*", "*.env", ".env.*"} for line in lines)

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert example.strip() == "XAI_API_KEY=your_key_here"
    assert FAKE_DOTENV not in example
    assert FAKE_SECRETS not in example

    ignored = subprocess.run(
        ["git", "check-ignore", "--", ".env", ".streamlit/secrets.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0
    ignored_paths = set(ignored.stdout.split())
    assert ".env" in ignored_paths
    assert ".streamlit/secrets.toml" in ignored_paths
    example_check = subprocess.run(
        ["git", "check-ignore", "-q", "--", ".env.example"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert example_check.returncode == 1
