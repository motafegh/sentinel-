"""Typed runtime-profile configuration for SENTINEL service processes.

This module owns environment classification and the dotenv/mock safety boundary.
It deliberately does not own scientific thresholds or service-specific settings;
those remain in their existing versioned configuration owners.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_RUNTIME_ENV_PREFIX = "SENTINEL_RUNTIME_"
_RUNTIME_ENV_KEYS = frozenset(
    {
        "SENTINEL_RUNTIME_CONFIG",
        "SENTINEL_RUNTIME_PROFILE",
        "SENTINEL_RUNTIME_DOTENV_MODE",
        "SENTINEL_RUNTIME_DOTENV_PATH",
        "SENTINEL_RUNTIME_MOCK_SERVICES",
    }
)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "runtime_default.yaml"


class RuntimeProfile(str, Enum):
    """The trust boundary under which a process is allowed to run."""

    TEST = "test"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class DotenvMode(str, Enum):
    """How an explicitly configured dotenv file may affect the environment."""

    DISABLED = "disabled"
    FILL_MISSING = "fill_missing"
    OVERRIDE = "override"


class DotenvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DotenvMode = DotenvMode.DISABLED
    path: str | None = None

    @model_validator(mode="after")
    def require_path_when_enabled(self) -> "DotenvConfig":
        if self.mode is not DotenvMode.DISABLED and not self.path:
            raise ValueError("dotenv.path is required when dotenv mode is enabled")
        return self


class RuntimeConfig(BaseModel):
    """Strict, versioned process safety configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    profile: RuntimeProfile = RuntimeProfile.DEVELOPMENT
    dotenv: DotenvConfig = Field(default_factory=DotenvConfig)
    mock_services: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def enforce_production_boundary(self) -> "RuntimeConfig":
        if self.profile is RuntimeProfile.PRODUCTION:
            if self.dotenv.mode is DotenvMode.OVERRIDE:
                raise ValueError("production profile forbids dotenv override")
            if self.mock_services:
                services = ", ".join(sorted(self.mock_services))
                raise ValueError(f"production profile forbids mock services: {services}")
        return self


_RUNTIME_CONFIG: RuntimeConfig | None = None


def load_runtime_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Load runtime configuration with overrides > environment > file precedence.

    Only the dedicated ``SENTINEL_RUNTIME_*`` namespace is inspected. Unknown
    names in that namespace are rejected so misspellings cannot silently select
    an unsafe profile.
    """

    source_env = os.environ if env is None else env
    _reject_unknown_runtime_env(source_env)

    resolved = _resolve_runtime_config_path(path, source_env)
    raw = _load_yaml_mapping(resolved) if resolved is not None else {}
    merged = _deep_merge(raw, _environment_overlay(source_env))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))
    return RuntimeConfig.model_validate(merged)


def get_runtime_config() -> RuntimeConfig:
    """Return the immutable process singleton."""

    global _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = load_runtime_config()
    return _RUNTIME_CONFIG


def reload_runtime_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Replace the process singleton; intended for service startup and tests."""

    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = load_runtime_config(path, env=env, overrides=overrides)
    return _RUNTIME_CONFIG


def apply_dotenv_policy(
    config: RuntimeConfig,
    *,
    loader: Callable[..., bool],
) -> bool:
    """Apply the validated dotenv policy through an injected dotenv loader.

    The caller injects ``python_dotenv.load_dotenv``. Injection keeps this
    boundary independently testable and prevents a dotenv read during import.
    """

    if config.dotenv.mode is DotenvMode.DISABLED:
        return False
    if config.profile is RuntimeProfile.PRODUCTION and config.dotenv.mode is DotenvMode.OVERRIDE:
        raise ValueError("production profile forbids dotenv override")

    dotenv_path = Path(config.dotenv.path or "").expanduser().resolve()
    if not dotenv_path.is_file():
        raise FileNotFoundError(f"Configured dotenv file not found: {dotenv_path}")
    return bool(
        loader(
            dotenv_path=dotenv_path,
            override=config.dotenv.mode is DotenvMode.OVERRIDE,
        )
    )


def bootstrap_environment(
    *,
    dotenv_path: str | Path | None = None,
    override: bool = True,
    env: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Centralised environment bootstrap for all SENTINEL entry points.

    1. Checks ``SENTINEL_ENV`` (raw env var, not runtime config — we need it
       to *load* the config).
    2. In production: does NOT load ``.env`` (must use real env vars).
    3. Non-production: loads ``.env`` via ``dotenv.load_dotenv``.
    4. Loads and returns the validated ``RuntimeConfig`` singleton.

    Call this once at the top of every ``main()`` or module-level startup
    **before** any service-specific ``os.getenv()`` or config reads.
    """
    source_env = os.environ if env is None else dict(env)
    sentinel_env = source_env.get("SENTINEL_ENV", "").lower()

    if sentinel_env != "production":
        from dotenv import load_dotenv as _load_dotenv

        resolved_path: Path | None = None
        if dotenv_path is not None:
            resolved_path = Path(dotenv_path).expanduser().resolve()
        if resolved_path is None or not resolved_path.is_file():
            # Walk up from CWD to find .env
            cwd = Path.cwd()
            for parent in [cwd, *cwd.parents]:
                candidate = parent / ".env"
                if candidate.is_file():
                    resolved_path = candidate
                    break
        if resolved_path is not None and resolved_path.is_file():
            _load_dotenv(dotenv_path=resolved_path, override=override)

    return reload_runtime_config(env=source_env)


def runtime_config_digest(config: RuntimeConfig) -> str:
    """Return a deterministic SHA-256 digest of the non-secret runtime config."""

    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_runtime_config_path(
    explicit: str | Path | None,
    env: Mapping[str, str],
) -> Path | None:
    candidate: str | Path | None = explicit or env.get("SENTINEL_RUNTIME_CONFIG")
    if candidate is not None:
        resolved = Path(candidate).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Runtime config file not found: {resolved}")
        return resolved
    return _DEFAULT_CONFIG_PATH if _DEFAULT_CONFIG_PATH.is_file() else None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Runtime config root must be a mapping: {path}")
    return loaded


def _environment_overlay(env: Mapping[str, str]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if profile := env.get("SENTINEL_RUNTIME_PROFILE"):
        overlay["profile"] = profile

    dotenv: dict[str, Any] = {}
    if mode := env.get("SENTINEL_RUNTIME_DOTENV_MODE"):
        dotenv["mode"] = mode
    if path := env.get("SENTINEL_RUNTIME_DOTENV_PATH"):
        dotenv["path"] = path
    if dotenv:
        overlay["dotenv"] = dotenv

    if "SENTINEL_RUNTIME_MOCK_SERVICES" in env:
        overlay["mock_services"] = [
            item.strip()
            for item in env["SENTINEL_RUNTIME_MOCK_SERVICES"].split(",")
            if item.strip()
        ]
    return overlay


def _reject_unknown_runtime_env(env: Mapping[str, str]) -> None:
    unknown = sorted(
        key for key in env if key.startswith(_RUNTIME_ENV_PREFIX) and key not in _RUNTIME_ENV_KEYS
    )
    if unknown:
        raise ValueError(f"Unknown SENTINEL runtime environment keys: {', '.join(unknown)}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "DotenvConfig",
    "DotenvMode",
    "RuntimeConfig",
    "RuntimeProfile",
    "apply_dotenv_policy",
    "bootstrap_environment",
    "get_runtime_config",
    "load_runtime_config",
    "reload_runtime_config",
    "runtime_config_digest",
]
