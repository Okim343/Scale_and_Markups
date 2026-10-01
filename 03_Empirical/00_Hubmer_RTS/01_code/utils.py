"""
Shared utilities for the Hubmer/GNR RTS estimation pipeline.

Provides:
  - PATHS         : canonical directories for the RTS pipeline.
  - load_config() : reads the local config.yaml (or an alternative passed
                    on the command line via --config).
  - setup_logger(): stage-specific logger writing to stdout.
  - ensure_dir()  : mkdir -p helper.
  - resolve_path(): resolve config paths relative to the project root.

This module is intentionally self-contained: it does NOT import anything
from 01_industry_alpha, so the RTS pipeline can run (and fail) without
touching the main alpha pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# This file lives at:
#   <PROJECT_ROOT>/03_Empirical/00_Hubmer_RTS/01_code/utils.py
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[3]
RTS_ROOT = _THIS.parents[1]               # 00_Hubmer_RTS
CODE = RTS_ROOT / "01_code"


@dataclass(frozen=True)
class _Paths:
    project_root: Path = PROJECT_ROOT
    rts_root: Path = RTS_ROOT
    code: Path = CODE

    @property
    def alpha_pipeline_root(self) -> Path:
        return self.project_root / "03_Empirical" / "01_industry_alpha"

    @property
    def alpha_salgado_dir(self) -> Path:
        """Integration point: where the alpha pipeline looks for RTS inputs."""
        return self.alpha_pipeline_root / "00_indata" / "04_salgado_data"


PATHS = _Paths()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the RTS config.yaml. Defaults to <01_code>/config.yaml."""
    if path is None:
        path = CODE / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def parse_config_arg(description: str) -> dict[str, Any]:
    """Standard CLI for all stage scripts: optional --config <path>."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an alternative config.yaml (e.g. config_smoke.yaml).",
    )
    args = parser.parse_args()
    return load_config(args.config)


def resolve_path(config: dict[str, Any], key_path: str) -> Path:
    """Resolve a config path entry relative to the project root.

    `key_path` is a dotted key, e.g. "input.source" or "output.intermediary_dir".
    Absolute paths in the config are returned unchanged. Relative paths are
    interpreted relative to PROJECT_ROOT.
    """
    node: Any = config
    for k in key_path.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"config key {key_path!r} not found (missing {k!r})")
        node = node[k]
    p = Path(str(node))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def intermediary_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config, "output.intermediary_dir")


def outdata_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config, "output.outdata_dir")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure a stage-level logger writing to stdout with timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Make sure a directory exists; return the path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
