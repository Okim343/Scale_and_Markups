"""Shared helpers for the Orbis Spain DLEU markup scripts (paths, config, logging, Stata pctile)."""

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE.parents[3]
DEFAULT_CONFIG = CODE / "config_markups_orbis_es.yaml"


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
    """(commit, branch, dirty) of the project repo; None entries when git is unavailable."""
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


def stata_pctile(x, p):
    """Percentile p (0-100) of x as Stata's `egen pctile()` (default _pctile definition).

    Missing values are ignored. With n sorted values and P = n * p / 100: if P is an integer,
    the mean of x_(P) and x_(P+1); otherwise x_(floor(P) + 1) (1-based order statistics).
    """
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    n = len(x)
    if n == 0:
        return np.nan
    num = n * p                       # P = num / 100, exact integer test for integer p
    if float(p).is_integer() and int(num) % 100 == 0:
        i = int(num) // 100
        if i <= 0:
            return x[0]
        if i >= n:
            return x[-1]
        return 0.5 * (x[i - 1] + x[i])
    i = int(np.floor(n * p / 100.0))
    return x[min(i, n - 1)]


def weighted_quantile(x, w, q):
    """Smallest x whose cumulative weight share reaches q (DLEU III.B: firms ranked by markup,
    weighted by sales share)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q, side="left").clip(max=len(x) - 1)]
