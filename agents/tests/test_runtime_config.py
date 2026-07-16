"""R0.0 tests for the typed runtime-profile safety boundary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from src.config.runtime import (
    DotenvMode,
    RuntimeProfile,
    apply_dotenv_policy,
    load_runtime_config,
    runtime_config_digest,
)


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "runtime.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_default_config_is_explicit_development_profile() -> None:
    config = load_runtime_config(env={})
    assert config.profile is RuntimeProfile.DEVELOPMENT
    assert config.dotenv.mode is DotenvMode.DISABLED
    assert config.mock_services == frozenset()


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, 'schema_version: "1"\nprofile: development\nrogue: true\n')
    with pytest.raises(ValidationError, match="rogue"):
        load_runtime_config(path, env={})


def test_unknown_runtime_environment_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="SENTINEL_RUNTIME_PROFLIE"):
        load_runtime_config(env={"SENTINEL_RUNTIME_PROFLIE": "production"})


def test_environment_overrides_file(tmp_path: Path) -> None:
    path = _write_config(tmp_path, 'schema_version: "1"\nprofile: development\n')
    config = load_runtime_config(
        path,
        env={"SENTINEL_RUNTIME_PROFILE": "test"},
    )
    assert config.profile is RuntimeProfile.TEST


def test_explicit_overrides_have_highest_precedence(tmp_path: Path) -> None:
    path = _write_config(tmp_path, 'schema_version: "1"\nprofile: development\n')
    config = load_runtime_config(
        path,
        env={"SENTINEL_RUNTIME_PROFILE": "test"},
        overrides={"profile": "production"},
    )
    assert config.profile is RuntimeProfile.PRODUCTION


def test_production_rejects_dotenv_override(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("EXAMPLE=value\n", encoding="utf-8")
    path = _write_config(
        tmp_path,
        (
            'schema_version: "1"\n'
            "profile: production\n"
            "dotenv:\n"
            "  mode: override\n"
            f"  path: {dotenv}\n"
        ),
    )
    with pytest.raises(ValidationError, match="forbids dotenv override"):
        load_runtime_config(path, env={})


def test_production_rejects_any_mock_service(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        ('schema_version: "1"\n' "profile: production\n" "mock_services:\n" "  - inference\n"),
    )
    with pytest.raises(ValidationError, match="forbids mock services: inference"):
        load_runtime_config(path, env={})


def test_development_can_explicitly_label_mock_services(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        ('schema_version: "1"\n' "profile: development\n" "mock_services:\n" "  - inference\n"),
    )
    config = load_runtime_config(path, env={})
    assert config.mock_services == frozenset({"inference"})


def test_enabled_dotenv_requires_a_path(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'schema_version: "1"\nprofile: development\ndotenv:\n  mode: fill_missing\n',
    )
    with pytest.raises(ValidationError, match="dotenv.path is required"):
        load_runtime_config(path, env={})


def test_disabled_dotenv_never_calls_loader() -> None:
    loader = Mock()
    config = load_runtime_config(env={})
    assert apply_dotenv_policy(config, loader=loader) is False
    loader.assert_not_called()


def test_fill_missing_dotenv_never_overrides_environment(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("EXAMPLE=value\n", encoding="utf-8")
    path = _write_config(
        tmp_path,
        (
            'schema_version: "1"\n'
            "profile: development\n"
            "dotenv:\n"
            "  mode: fill_missing\n"
            f"  path: {dotenv}\n"
        ),
    )
    loader = Mock(return_value=True)
    config = load_runtime_config(path, env={})
    assert apply_dotenv_policy(config, loader=loader) is True
    loader.assert_called_once_with(dotenv_path=dotenv.resolve(), override=False)


def test_missing_configured_dotenv_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"
    path = _write_config(
        tmp_path,
        (
            'schema_version: "1"\n'
            "profile: development\n"
            "dotenv:\n"
            "  mode: fill_missing\n"
            f"  path: {missing}\n"
        ),
    )
    config = load_runtime_config(path, env={})
    with pytest.raises(FileNotFoundError, match="Configured dotenv file not found"):
        apply_dotenv_policy(config, loader=Mock())


def test_runtime_config_digest_is_stable_and_sensitive() -> None:
    development = load_runtime_config(env={})
    development_again = load_runtime_config(env={})
    test_profile = load_runtime_config(
        env={"SENTINEL_RUNTIME_PROFILE": "test"},
    )
    assert runtime_config_digest(development) == runtime_config_digest(development_again)
    assert runtime_config_digest(development) != runtime_config_digest(test_profile)
