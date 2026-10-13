"""Shared helpers for the Orbis Spain moments scripts (paths, config, logging, provenance)."""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE.parents[3]
DEFAULT_CONFIG = CODE / "config_moments_orbis_es.yaml"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(path.read_text()), path


def setup_logger(name, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def git_state():
    """(commit, branch, dirty) of the project repo; dirty refers to this code folder."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            return None
    status = git("status", "--porcelain", "--", str(CODE.relative_to(PROJECT_ROOT)))
    return git("rev-parse", "HEAD") or None, git("rev-parse", "--abbrev-ref", "HEAD") or None, bool(status)


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    """Provenance of an input or output file: project-relative path, sha256, size, parquet rows."""
    path = Path(path)
    fp = {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "sha256": file_sha256(path),
          "bytes": path.stat().st_size}
    if path.suffix == ".parquet":
        fp["rows"] = pq.ParquetFile(path).metadata.num_rows
    return fp


def dirs(config):
    """(intermediary, outdata) directories, created if missing."""
    inter = resolve(config["output"]["intermediary_dir"])
    outd = resolve(config["output"]["outdata_dir"])
    inter.mkdir(parents=True, exist_ok=True)
    outd.mkdir(parents=True, exist_ok=True)
    return inter, outd


def write_json(path, obj):
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(obj, indent=2, default=default))
