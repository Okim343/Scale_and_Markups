"""Calibrated-parameter file: read, write and provenance.

Handoff between ``04_Estimation`` (writer) and ``05_Simulation`` (reader).
Ported from ``04_Code/steady_state/__main__.py`` (``_load_calibration_yaml``,
``_write_yaml``, ``_native``), with a provenance block added. The file holds:

* ``parameters``: the calibrated ``(xi, N, gamma, eta, rho_bar)``, the anchored
  scale ``v_min`` and anchor ``y_hat``, and the assigned ``a, H, phi_v``;
* ``reference``: the GE prices of the anchored solve (``w, R, X_market, ...``),
  used as the initial condition of the frozen-technology GE solves downstream;
* free-form blocks written by the estimation (``entry``, ``fit``, ...);
* ``provenance``: the sha256 of the bundle's ``aggregate_moments.yaml`` and
  ``F_alpha.parquet`` (``bundle_sha256``), plus the bundle manifest sha256,
  config sha256s, git commit and dirty flag.

:func:`load_params_file` refuses a file whose ``bundle_sha256`` differs from
the bundle it is used with.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import yaml

from .model.normalization import PooledParams
from .primitives import ModelPrimitives, load_primitives


SCHEMA = "orbis-params-v1"
HASHED_BUNDLE_FILES = ("aggregate_moments.yaml", "F_alpha.parquet")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            h.update(block)
    return h.hexdigest()


def bundle_hashes(bundle_dir: Path | str) -> dict[str, str]:
    """sha256 of the bundle files that define the model inputs and targets."""
    bundle_dir = Path(bundle_dir)
    return {name: file_sha256(bundle_dir / name) for name in HASHED_BUNDLE_FILES}


def git_state(code_dirs: tuple[Path | str, ...] = ()) -> dict[str, Any]:
    """Commit, branch and dirty flag of the repo; dirty refers to ``code_dirs``."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
            )
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    paths = [str(Path(p).resolve().relative_to(PROJECT_ROOT)) for p in code_dirs]
    status = git("status", "--porcelain", "--", *paths) if paths else git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty": bool(status) if status is not None else None,
    }


def _relative(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_provenance(
    *,
    bundle_dir: Path | str,
    configs: Mapping[str, Path | str],
    code_dirs: tuple[Path | str, ...],
    script: str,
) -> dict[str, Any]:
    """Provenance block: bundle hashes, config hashes, git state, timestamp."""
    bundle_dir = Path(bundle_dir)
    manifest = bundle_dir / "manifest.yaml"
    if not manifest.exists():
        raise FileNotFoundError(f"Bundle manifest missing: {manifest}")
    return {
        "script": script,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_dir": _relative(bundle_dir),
        "bundle_sha256": bundle_hashes(bundle_dir),
        "bundle_manifest_sha256": file_sha256(manifest),
        "configs": {
            name: {"path": _relative(path), "sha256": file_sha256(path)}
            for name, path in configs.items()
        },
        **git_state(code_dirs),
    }


def native(item):
    """Convert numpy scalars and tuples so ``yaml.safe_dump`` accepts the payload."""
    if isinstance(item, dict):
        return {key: native(val) for key, val in item.items()}
    if isinstance(item, (list, tuple)):
        return [native(val) for val in item]
    if isinstance(item, np.generic):
        return item.item()
    return item


def write_yaml(path: Path | str, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        yaml.safe_dump(native(dict(value)), stream, sort_keys=False)


def params_payload(params: PooledParams) -> dict[str, float]:
    return {
        "xi": params.xi, "N": params.N,
        "gamma": params.gamma, "eta": params.eta,
        "rho_bar": params.rho_bar,
        "v_min": params.v_min,
        "y_hat": params.y_hat,
        "a": params.a, "H": params.H, "phi_v": params.phi_v,
    }


def write_params_file(
    path: Path | str,
    params: PooledParams,
    *,
    reference: Mapping[str, float],
    provenance: Mapping[str, Any],
    **blocks: Mapping[str, Any],
) -> Path:
    """Write the params file; extra keyword blocks (``fit``, ``entry``, ...) are kept verbatim."""
    if "bundle_sha256" not in provenance:
        raise ValueError("provenance must carry bundle_sha256")
    payload = {
        "schema": SCHEMA,
        "parameters": params_payload(params),
        "reference": dict(reference),
        **{name: dict(block) for name, block in blocks.items()},
        "provenance": dict(provenance),
    }
    write_yaml(path, payload)
    return Path(path)


def check_bundle_hashes(data: Mapping[str, Any], bundle_dir: Path | str, *, source: str = "params file") -> None:
    """Raise unless the file's ``provenance.bundle_sha256`` matches ``bundle_dir``."""
    recorded = (data.get("provenance") or {}).get("bundle_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{source} has no provenance.bundle_sha256; refusing to run")
    current = bundle_hashes(bundle_dir)
    bad = [name for name in HASHED_BUNDLE_FILES if recorded.get(name) != current[name]]
    if bad:
        raise ValueError(
            f"{source} was produced on a different bundle: sha256 mismatch for {bad} "
            f"(bundle {bundle_dir}); refusing to run"
        )


def load_params_file(
    path: Path | str, *, bundle_dir: Path | str,
) -> tuple[PooledParams, float | None, tuple[float, float] | None, dict[str, Any]]:
    """Load a params file after checking it was produced on ``bundle_dir``.

    Returns ``(params, wf_reference, initial, data)`` where ``initial`` is the
    ``(w, X_market)`` GE starting point from the ``reference`` block (as in
    04_Code's ``_load_calibration_yaml``) and ``data`` is the parsed file.
    """
    with open(path) as stream:
        data = yaml.safe_load(stream)
    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported params schema {schema!r}; expected {SCHEMA!r}")
    check_bundle_hashes(data, bundle_dir, source=str(path))
    values = data["parameters"]
    if any(isinstance(values.get(key), dict) for key in ("xi", "N", "gamma", "eta")):
        raise ValueError("params file must contain common scalar parameters")
    params = PooledParams(
        xi=float(values["xi"]), N=float(values["N"]),
        gamma=float(values["gamma"]), eta=float(values["eta"]),
        v_min=float(values["v_min"]), y_hat=float(values["y_hat"]),
        rho_bar=float(values["rho_bar"]),
        a=float(values["a"]), H=int(values["H"]), phi_v=float(values["phi_v"]),
    )
    reference = data.get("reference", {}) or {}
    wf = reference.get("wf_reference")
    initial = (
        (float(reference["w"]), float(reference["X_market"]))
        if "w" in reference and "X_market" in reference else None
    )
    return params, (float(wf) if wf is not None else None), initial, data


def load_simulation_inputs(
    cfg, params_path: Path | str, logger=None,
) -> tuple[ModelPrimitives, PooledParams, float | None, tuple[float, float] | None]:
    """Bundle primitives plus the hash-checked params file, for the 05 scripts.

    ``cfg`` is a :class:`orbis_model.config.SimulationConfig`. Returns
    ``(primitives, params, wf_reference, initial)``.
    """
    primitives = load_primitives(cfg.bundle_dir, H=cfg.parameters.H)
    params, wf, initial, _ = load_params_file(params_path, bundle_dir=cfg.bundle_dir)
    if (params.a, params.phi_v) != (primitives.a, primitives.phi_v):
        raise ValueError("params file a/phi_v differ from the bundle")
    if logger is not None:
        if params.H != primitives.H:
            logger.warning(f"params file H={params.H} differs from config H={primitives.H}; "
                           "using the params-file H")
        logger.info(f"params {params_path}: xi={params.xi:.6g} N={params.N:.6g} "
                    f"gamma={params.gamma:.6g} eta={params.eta:.6g} rho_bar={params.rho_bar:.6g} "
                    f"v_min={params.v_min:.6g} H={params.H}")
    return primitives, params, wf, initial
