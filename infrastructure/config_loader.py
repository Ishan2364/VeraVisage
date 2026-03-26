"""
infrastructure/config_loader.py
────────────────────────────────────────────────────────────────────────────
Reads and merges all YAML config files into one validated settings dict.

WHY THIS EXISTS
───────────────
Every threshold, path, and hyperparameter in this project lives in
configs/*.yaml — never hardcoded inside a .py file. This module is the
single bridge between those YAML files and the running Python code.

Any module that needs a config value does:

    from infrastructure.config_loader import load_config
    cfg = load_config()

    threshold = cfg["verification"]["accept_threshold"]   # 0.70
    device    = cfg["device"]                             # "cpu"

WHAT IT MERGES
──────────────
  base_config.yaml   → paths, device, seed, project name
  model_config.yaml  → all ML hyperparameters and thresholds

The result is ONE flat dict you can navigate with standard key access.

CACHING
───────
The merged config is cached after the first load. Subsequent calls to
load_config() return the same object instantly — no file I/O on every call.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from infrastructure.logger import get_logger

log = get_logger(__name__)

# ── Module-level cache ────────────────────────────────────────────────────
# None means "not yet loaded". After the first load_config() call this holds
# the merged dict so repeated calls are free.
_config_cache: dict[str, Any] | None = None

# YAML files to merge, in order. Later files override earlier ones if a key
# appears in both — so model_config can override a base_config default.
_CONFIG_FILES = ["base_config.yaml", "model_config.yaml"]


def load_config(
    config_dir: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """
    Load and merge all project YAML configs into one dict.

    Args:
        config_dir:   Path to the configs/ folder. Defaults to
                      <project_root>/configs/ auto-detected from this file's
                      location — you almost never need to pass this.
        force_reload: Set True to bypass the cache and re-read from disk.
                      Useful in tests that swap config files.

    Returns:
        Merged configuration dict. Top-level keys match the YAML structure:
            cfg["device"]
            cfg["paths"]["raw_data"]
            cfg["verification"]["accept_threshold"]
            cfg["liveness"]["aggregator_weights"]["texture"]
            ... etc.

    Raises:
        FileNotFoundError: If a required config file is missing.
        yaml.YAMLError:    If a config file contains invalid YAML.
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    # ── Resolve the configs directory ─────────────────────────────────────
    if config_dir is None:
        # This file lives at <project_root>/infrastructure/config_loader.py
        # so parents[1] is <project_root>
        config_dir = Path(__file__).resolve().parents[1] / "configs"
    else:
        config_dir = Path(config_dir)

    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"Config directory not found: {config_dir}\n"
            f"Have you run generate_structure.py yet?"
        )

    # ── Merge all YAML files ───────────────────────────────────────────────
    merged: dict[str, Any] = {}

    for filename in _CONFIG_FILES:
        file_path = config_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required config file missing: {file_path}\n"
                f"Expected files: {_CONFIG_FILES}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            log.warning("Config file is empty, skipping: %s", file_path)
            continue

        # Deep merge: for nested dicts, merge recursively rather than
        # overwriting the entire sub-dict when a key appears in both files.
        _deep_merge(merged, data)
        log.debug("Loaded config: %s", file_path.name)

    log.info(
        "Configuration loaded — device=%s, seed=%s",
        merged.get("device", "unknown"),
        merged.get("seed", "unknown"),
    )

    _config_cache = merged
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """
    Recursively merge `override` into `base` in-place.

    For nested dicts, keys are merged at every level rather than the top-
    level key being replaced wholesale. For all other types (str, int, list)
    the override value wins.

    Example:
        base     = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 99, "z": 3}}
        result   → {"a": {"x": 1, "y": 99, "z": 3}}   ← x preserved

    Without deep merge you'd lose "x" because the entire "a" dict would be
    replaced by the override's "a" dict.
    """
    for key, override_val in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(override_val, dict)
        ):
            _deep_merge(base[key], override_val)
        else:
            base[key] = override_val