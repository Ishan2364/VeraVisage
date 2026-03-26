"""
infrastructure/logger.py
────────────────────────────────────────────────────────────────────────────
Centralised logger factory for the entire deepfake_auth project.

RULE: No module in this project ever calls print() or creates its own
logging.getLogger() directly. Every module does exactly this:

    from infrastructure.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")

This gives every log line:
  - A timestamp
  - The severity level
  - The exact module that emitted it  ← priceless when debugging
  - The message

Logs go to TWO places simultaneously:
  1. The terminal (console), so you see output while running
  2. logs/deepfake_auth.log on disk, so you can review history
────────────────────────────────────────────────────────────────────────────
"""

import logging
import logging.config
from pathlib import Path

import yaml


# ── Internal state ────────────────────────────────────────────────────────
# We only load and apply the config once, no matter how many modules call
# get_logger(). This flag tracks whether we've done that initialisation.
_logging_configured = False


def _configure_logging() -> None:
    """
    Reads configs/logging_config.yaml and applies it via dictConfig.

    Called automatically the first time get_logger() is invoked.
    If the config file is missing we fall back to a safe basicConfig so
    the project still runs — it just won't write to a log file.
    """
    global _logging_configured

    if _logging_configured:
        return  # Already done — don't reconfigure on every import

    # Walk up from this file's location to find the project root,
    # then look for configs/logging_config.yaml
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "logging_config.yaml"
    logs_dir = project_root / "logs"

    # Ensure the logs directory exists before the FileHandler tries to open it
    logs_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            log_cfg = yaml.safe_load(f)

        # Patch the FileHandler filename to use an absolute path so the log
        # file is always created at <project_root>/logs/deepfake_auth.log
        # regardless of where the user runs the script from.
        for handler in log_cfg.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(logs_dir / Path(handler["filename"]).name)

        logging.config.dictConfig(log_cfg)
    else:
        # Fallback: no config file found — use a simple console-only setup
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.warning(
            "logging_config.yaml not found at %s — using fallback config.",
            config_path,
        )

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a fully configured Logger for the given name.

    Usage (copy-paste this into every new module):

        from infrastructure.logger import get_logger
        log = get_logger(__name__)

    Then use:
        log.debug("Verbose detail only needed when debugging")
        log.info("Normal operational messages")
        log.warning("Something unexpected but recoverable")
        log.error("Something failed — needs attention")
        log.exception("Inside an except block — auto-attaches traceback")

    Args:
        name: Conventionally pass __name__ so the log line shows the
              exact module path, e.g. "core_vision.face_detector".

    Returns:
        A standard library Logger instance, already configured.
    """
    _configure_logging()
    return logging.getLogger(name)