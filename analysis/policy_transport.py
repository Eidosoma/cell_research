"""E04 S12 policy-specific transport extraction and predictive modelling.

The design is frozen in ``analysis/s12_policy_transport_contract.json``.
Stages are deliberately separated so external trajectories are not decoded until
the selected development-only model has been serialized and hashed.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from reference_simulator.model import canonical_json_bytes


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
OUTPUT = Path("/artifacts/research_steps/S12")
CACHE = Path("/cache/e04_s12")
CONTRACT = REPOSITORY / "analysis/s12_policy_transport_contract.json"
S07_CACHE = Path("/cache/e04_s07/holdout")
S03_MANIFEST = Path("/artifacts/research_steps/S03/scenario_manifest.parquet")
S13_DIR = Path("/artifacts/research_steps/S13")
POLICIES = ("Bubble", "Insertion", "Selection")
POLICY_CODES = {name: index for index, name in enumerate(POLICIES)}
REGIMES = (
    "native_control",
    "activation_equal",
    "displacement_equal",
    "target_range_equal",
    "success_equal",
    "joint_equal",
)
FIT_REPLICATES = (
    15,
    25,
    35,
    45,
    65,
    75,
    85,
    95,
    115,
    125,
    135,
    145,
    165,
    175,
    185,
    195,
    215,
    225,
    235,
    245,
)
CALIBRATION_REPLICATES = (5, 55, 105, 155, 205)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FOREST_CANDIDATES = ((8, 5), (14, 5), (None, 5), (None, 15))
PCA_COMPONENTS = 12
GRID = np.linspace(0.0, 1.0, 101)
EXPECTED_SCENARIOS = 4650
EXPECTED_ROWS = EXPECTED_SCENARIOS * len(REGIMES)
BOOTSTRAP_SEED = 20260717


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_native(value))).hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_native(value)) + b"\n")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def _decode(value: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(value), dtype=dtype)
    if raw.size != math.prod(shape):
        raise ValueError(f"encoded shape mismatch: {raw.size} != {math.prod(shape)}")
    return raw.reshape(shape).copy()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _verify_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    return {
        "directory": str(directory),
        "manifestSha256": sha256_file(manifest_path),
        "artifactCount": len(checks),
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S12 frozen policy-transport specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S12 |
| Completion status | Design frozen before model fitting or held-out outcome decoding |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Upstream S01–S11R manifests and all listed files rehashed; S07 raw corpus present; S13 absent; split and model-family structure passed |
| Outcome classification | Pending S12 execution |
| Caveats or blockers | The predictors are complete-run coarse summaries. Collision load is an operational no-op/blockage measure and residence is 1%-grid interval-censored. S11R infeasibility remains a mechanistic constraint. |
| Recommended next action | Fit on the frozen native development partition, hash the selected model, then evaluate sealed native/intervention strata; do not start S13 |

Contract SHA-256: `{contract_hash}`.

## Frozen estimand and statistics

The target is the 101-point composition-corrected publication-adjacency
trajectory. For each executable policy, the transport vector contains actor
velocity, blocked-action collision load, experienced positional flux, and
grid-residence time. The first three are derived from S07's exact event ledger;
residence is derived from identity positions on the accepted-swap grid.

This is retrospective trajectory prediction from summaries of a completed run,
not online forecasting. A strong fit can show that the summaries are sufficient
for prediction in tested regimes; it cannot identify a unique causal mechanism.
S11 descriptive Shapley estimates are neither inputs nor validation targets.

## Frozen partitions

Development contains only native, absent-association interior pairwise
compositions (20/80 through 80/20) and balanced three-way mixtures. Five fixed
replicate ordinals per condition calibrate hyperparameters and conformal bands;
twenty fit them. Native 10/90 and 90/10 tails, rare three-way mixtures, all
positive/negative policy-value constructions, and every kinetic intervention
are held out. Only `activation_equal` is a feasible confirmatory intervention;
the other S07 regimes are out-of-distribution stress tests, not matched causal
controls.

## Models and decision

Simple baselines are a shifted mean trajectory and static-covariate PCA ridge.
Candidate transport families are PCA ridge, PCA ExtraTrees, and a recursively
rolled first-order ridge Markov model. Selection uses development-calibration
condition-mean RMSE only. The selected family is refit on all development rows,
serialized, and hashed before external trajectories are decoded. Frozen native
and activation-equal accuracy, improvement, endpoint, bias, and coverage gates
determine the outcome; residual and failure-region audits qualify it.
"""


def freeze_design(output: Path = OUTPUT, cache: Path = CACHE) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    if S13_DIR.exists():
        raise AssertionError("S13 artifacts exist before S12 freeze")
    contract_hash = sha256_file(CONTRACT)
    upstream = {}
    for step in [f"S{i:02d}" for i in range(1, 12)] + ["S11R"]:
        upstream[step] = _verify_manifest(Path("/artifacts/research_steps") / step)
    if not all(item["passed"] for item in upstream.values()):
        raise AssertionError("upstream immutability failed before freeze")
    raw = {}
    for regime in REGIMES:
        path = S07_CACHE / f"{regime}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw[regime] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = pd.read_parquet(
        S03_MANIFEST, columns=["scenario_id", "condition_id", "replicate_ordinal"]
    )
    holdout = manifest[
        manifest.replicate_ordinal.isin(FIT_REPLICATES + CALIBRATION_REPLICATES)
    ]
    if (
        len(holdout) != EXPECTED_SCENARIOS
        or holdout.scenario_id.nunique() != EXPECTED_SCENARIOS
    ):
        raise AssertionError("S07/S03 source population changed")
    record = {
        "schema": "e04.s12.freeze_record.v1",
        "researchStepId": "S12",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeFitting": True,
        "externalOutcomesOpened": False,
        "contractPath": str(CONTRACT),
        "contractSha256": contract_hash,
        "fitReplicateOrdinals": FIT_REPLICATES,
        "calibrationReplicateOrdinals": CALIBRATION_REPLICATES,
        "replicateOverlap": sorted(set(FIT_REPLICATES) & set(CALIBRATION_REPLICATES)),
        "sourceScenarioCount": len(holdout),
        "rawCorpus": raw,
        "upstream": upstream,
        "s13Absent": not S13_DIR.exists(),
    }
    if record["replicateOverlap"]:
        raise AssertionError("fit/calibration overlap")
    write_json(output / "preregistration.json", json.loads(CONTRACT.read_text()))
    write_json(output / "freeze_record.json", record)
    (output / "transport_model_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen() -> dict[str, Any]:
    record = json.loads((OUTPUT / "freeze_record.json").read_text())
    if record["contractSha256"] != sha256_file(CONTRACT):
        raise AssertionError("contract changed after freeze")
    if not record["frozenBeforeFitting"] or record["replicateOverlap"]:
        raise AssertionError("invalid S12 freeze")
    if S13_DIR.exists():
        raise AssertionError("S13 artifacts appeared during S12")
    return record


def _scenario_covariates() -> dict[str, dict[str, float]]:
    frame = pd.read_parquet(
        S03_MANIFEST,
        columns=["scenario_id", "achieved_spearman_rho", "achieved_eta_squared"],
    )
    return {
        str(row.scenario_id): {
            "achieved_spearman_rho": float(row.achieved_spearman_rho),
            "achieved_eta_squared": float(row.achieved_eta_squared),
        }
        for row in frame.itertuples(index=False)
    }


def classify_split(row: Mapping[str, Any]) -> str:
    regime = str(row["regime"])
    if regime != "native_control":
        return (
            "intervention_activation_equal"
            if regime == "activation_equal"
            else "intervention_infeasible_sensitivity"
        )
    correlation = str(row["correlation_profile"])
    profile = str(row["composition_profile"])
    if correlation != "absent":
        return "native_association_holdout"
    interior_pair = str(row["composition_class"]) == "pairwise" and int(
        row["first_policy_count"]
    ) in range(20, 81, 10)
    balanced_three = profile == "balanced_rotating"
    if interior_pair or balanced_three:
        replicate = int(row["replicate_ordinal"])
        if replicate in FIT_REPLICATES:
            return "development_fit"
        if replicate in CALIBRATION_REPLICATES:
            return "development_calibration"
        raise AssertionError(f"unexpected development replicate {replicate}")
    return "native_composition_holdout"


def _residence(occupancy: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positions = np.argsort(occupancy, axis=1)
    moves = np.count_nonzero(positions[1:] != positions[:-1], axis=0)
    identity_residence = 100.0 / (moves + 1.0)
    return {
        policy: float(np.mean(identity_residence[labels == code]))
        if np.any(labels == code)
        else 0.0
        for policy, code in POLICY_CODES.items()
    }


def extract_run(
    row: Mapping[str, Any], covariates: Mapping[str, float], *, validate: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    occupancy = _decode(str(row["occupancy_b64"]), np.dtype(np.uint8), (101, 100))
    labels = _decode(str(row["labels_b64"]), np.dtype(np.uint8), (100,))
    curve = _decode(str(row["corrected_curve_b64"]), np.dtype(np.float64), (101,))
    kinetics = json.loads(str(row["policy_kinetics_json"]))
    counts = json.loads(str(row["policy_counts_json"]))
    residence = _residence(occupancy, labels)
    ledger_rows = []
    validation = {
        "scenario_id": row["scenario_id"],
        "regime": row["regime"],
        "occupancy_permutation_passed": True,
        "label_count_passed": True,
        "curve_max_error": 0.0,
        "structural_identity_max_error": 0.0,
        "activation_accounting_max_error": 0.0,
        "flux_conservation_error": 0.0,
        "residence_bounds_passed": True,
    }
    if validate:
        expected = np.arange(100, dtype=np.uint8)
        validation["occupancy_permutation_passed"] = bool(
            all(np.array_equal(np.sort(state), expected) for state in occupancy)
        )
        observed_counts = {
            policy: int(np.count_nonzero(labels == code))
            for policy, code in POLICY_CODES.items()
            if np.any(labels == code)
        }
        validation["label_count_passed"] = observed_counts == {
            key: int(value) for key, value in counts.items()
        }
        state_labels = labels[occupancy]
        same = np.count_nonzero(state_labels[:, 1:] == state_labels[:, :-1], axis=1)
        recalculated = same / 100.0 - float(row["publication_aggregation_null"])
        validation["curve_max_error"] = float(np.max(np.abs(curve - recalculated)))
    total_exp = 0.0
    total_actor_range = 0.0
    for policy in POLICIES:
        values = kinetics.get(policy)
        present = values is not None
        if present:
            actor = float(values["actor_activations"])
            accepted = float(values["accepted_swaps"])
            no_ops = float(values["no_ops"])
            rejected = float(values["gate_rejections"])
            memory = float(values["memory_updates"])
            velocity = float(values["actor_displacement_rate"] or 0.0)
            collision = (no_ops + rejected) / actor if actor else 0.0
            flux = float(values["experienced_displacement_rate"] or 0.0)
            structural = abs(
                velocity
                - (accepted / actor if actor else 0.0)
                * float(values["executed_target_range"] or 0.0)
            )
            accounting = abs(actor - no_ops - rejected - memory - accepted)
            total_exp += float(values["experienced_displacement_sum"])
            total_actor_range += float(values["accepted_range_sum"])
            validation["structural_identity_max_error"] = max(
                validation["structural_identity_max_error"], structural
            )
            validation["activation_accounting_max_error"] = max(
                validation["activation_accounting_max_error"], accounting
            )
        else:
            actor = accepted = no_ops = rejected = memory = 0.0
            velocity = collision = flux = 0.0
        residence_value = residence[policy]
        if not (100.0 / 101.0 - 1e-12 <= residence_value <= 100.0 + 1e-12) and present:
            validation["residence_bounds_passed"] = False
        ledger_rows.append(
            {
                "schema_version": "e04.s12.transport_statistics.v1",
                "research_step_id": "S12",
                "scenario_id": row["scenario_id"],
                "condition_id": row["condition_id"],
                "replicate_ordinal": int(row["replicate_ordinal"]),
                "regime": row["regime"],
                "split": classify_split(row),
                "input_profile": row["input_profile"],
                "policy_set_label": row["policy_set_label"],
                "composition_profile": row["composition_profile"],
                "correlation_profile": row["correlation_profile"],
                "policy": policy,
                "policy_present": present,
                "policy_count": int(counts.get(policy, 0)),
                "actor_velocity": velocity,
                "collision_load": collision,
                "experienced_flux": flux,
                "residence_time": residence_value,
                "actor_activations": actor,
                "accepted_swaps": accepted,
                "no_ops": no_ops,
                "gate_rejections": rejected,
                "memory_updates": memory,
            }
        )
    validation["flux_conservation_error"] = abs(total_exp - 2.0 * total_actor_range)
    features: dict[str, Any] = {
        "scenario_id": row["scenario_id"],
        "condition_id": row["condition_id"],
        "replicate_ordinal": int(row["replicate_ordinal"]),
        "regime": row["regime"],
        "split": classify_split(row),
        "input_profile": row["input_profile"],
        "policy_set_label": row["policy_set_label"],
        "composition_class": row["composition_class"],
        "composition_profile": row["composition_profile"],
        "correlation_profile": row["correlation_profile"],
        "initial_corrected": float(curve[0]),
        "unique_input": float(str(row["input_profile"]).startswith("unique")),
        "achieved_spearman_rho": float(covariates["achieved_spearman_rho"]),
        "achieved_eta_squared": float(covariates["achieved_eta_squared"]),
        "activation_count": int(row["activation_count"]),
        "successful_swap_count": int(row["successful_swap_count"]),
        "completed": bool(row["completed"]),
    }
    for item in ledger_rows:
        suffix = str(item["policy"]).lower()
        features[f"fraction_{suffix}"] = float(item["policy_count"]) / 100.0
        features[f"present_{suffix}"] = float(item["policy_present"])
        for metric in (
            "actor_velocity",
            "collision_load",
            "experienced_flux",
            "residence_time",
        ):
            features[f"{metric}_{suffix}"] = float(item[metric])
    features["curve"] = curve
    return features, ledger_rows, validation


STATIC_NAMES = [
    "initial_corrected",
    "fraction_bubble",
    "fraction_insertion",
    "fraction_selection",
    "present_bubble",
    "present_insertion",
    "present_selection",
    "unique_input",
    "achieved_spearman_rho",
    "achieved_eta_squared",
]
TRANSPORT_NAMES = [
    f"{metric}_{policy.lower()}"
    for metric in (
        "actor_velocity",
        "collision_load",
        "experienced_flux",
        "residence_time",
    )
    for policy in POLICIES
]


def feature_matrix(rows: Sequence[Mapping[str, Any]], transport: bool) -> np.ndarray:
    names = STATIC_NAMES + (TRANSPORT_NAMES if transport else [])
    return np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=float
    )


def curve_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(row["curve"], dtype=float) for row in rows])


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_development() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    assert_frozen()
    covariates = _scenario_covariates()
    fit: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for raw in _read_jsonl(S07_CACHE / "native_control.jsonl"):
        split = classify_split(raw)
        if not split.startswith("development"):
            continue
        row, _, check = extract_run(raw, covariates[str(raw["scenario_id"])])
        (fit if split == "development_fit" else calibration).append(row)
        validation.append(check)
    return fit, calibration, validation


def _condition_mean_rmse(
    rows: Sequence[Mapping[str, Any]], observed: np.ndarray, predicted: np.ndarray
) -> float:
    by_condition: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_condition[str(row["condition_id"])].append(index)
    errors = []
    for indices in by_condition.values():
        errors.append(
            np.mean(
                (observed[indices].mean(axis=0) - predicted[indices].mean(axis=0)) ** 2
            )
        )
    return float(np.sqrt(np.mean(errors)))


def _condition_metrics(
    rows: Sequence[Mapping[str, Any]], observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    by_condition: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_condition[str(row["condition_id"])].append(index)
    rmses, maes, peaks, areas = [], [], [], []
    for indices in by_condition.values():
        obs = observed[indices].mean(axis=0)
        pred = predicted[indices].mean(axis=0)
        rmses.append(float(np.sqrt(np.mean((obs - pred) ** 2))))
        maes.append(float(np.mean(np.abs(obs - pred))))
        peaks.append(abs(float(np.max(obs) - np.max(pred))))
        areas.append(
            abs(
                float(
                    np.trapezoid(np.maximum(obs, 0.0), GRID)
                    - np.trapezoid(np.maximum(pred, 0.0), GRID)
                )
            )
        )
    return {
        "condition_rmse_mean": float(np.mean(rmses)),
        "pooled_rmse": _condition_mean_rmse(rows, observed, predicted),
        "condition_mae_mean": float(np.mean(maes)),
        "peak_mae": float(np.mean(peaks)),
        "positive_area_mae": float(np.mean(areas)),
    }


def _pca_fit(y: np.ndarray) -> PCA:
    residual = y - y[:, [0]]
    return PCA(n_components=PCA_COMPONENTS, random_state=BOOTSTRAP_SEED).fit(residual)


def _fit_pca_ridge(
    x: np.ndarray, y: np.ndarray, pca: PCA, alpha: float
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), pca.transform(y - y[:, [0]]))
    return {"family": "pca_ridge", "scaler": scaler, "regressor": model, "pca": pca}


def _fit_pca_forest(
    x: np.ndarray, y: np.ndarray, pca: PCA, depth: int | None, leaf: int
) -> dict[str, Any]:
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=depth,
        min_samples_leaf=leaf,
        max_features=1.0,
        n_jobs=8,
        random_state=BOOTSTRAP_SEED,
    ).fit(x, pca.transform(y - y[:, [0]]))
    return {"family": "pca_forest", "regressor": model, "pca": pca}


def _markov_design(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    repeated = np.repeat(x, 100, axis=0)
    current = y[:, :-1].reshape(-1, 1)
    p = np.tile(GRID[:-1], len(y)).reshape(-1, 1)
    design = np.hstack(
        [repeated, current, p, p**2, np.sin(np.pi * p), np.cos(np.pi * p)]
    )
    target = (y[:, 1:] - y[:, :-1]).reshape(-1)
    return design, target


def _fit_markov(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    design, target = _markov_design(x, y)
    scaler = StandardScaler().fit(design)
    model = Ridge(alpha=alpha).fit(scaler.transform(design), target)
    return {"family": "markov", "scaler": scaler, "regressor": model}


def predict_model(
    model: Mapping[str, Any], x: np.ndarray, y0: np.ndarray
) -> np.ndarray:
    family = str(model["family"])
    if family == "mean_shape":
        prediction = y0[:, None] + np.asarray(model["mean_residual"])[None, :]
    elif family == "pca_ridge":
        scores = model["regressor"].predict(model["scaler"].transform(x))
        prediction = y0[:, None] + model["pca"].inverse_transform(scores)
    elif family == "pca_forest":
        scores = model["regressor"].predict(x)
        prediction = y0[:, None] + model["pca"].inverse_transform(scores)
    elif family == "markov":
        prediction = np.empty((len(x), 101), dtype=float)
        prediction[:, 0] = y0
        for index, progress in enumerate(GRID[:-1]):
            p = np.full((len(x), 1), progress)
            design = np.hstack(
                [
                    x,
                    prediction[:, [index]],
                    p,
                    p**2,
                    np.sin(np.pi * p),
                    np.cos(np.pi * p),
                ]
            )
            delta = model["regressor"].predict(model["scaler"].transform(design))
            prediction[:, index + 1] = prediction[:, index] + delta
    else:
        raise ValueError(f"unknown family {family}")
    prediction[:, 0] = y0
    return np.clip(prediction, -1.0, 1.0)


def _candidate_name(family: str, parameters: Mapping[str, Any]) -> str:
    values = "_".join(f"{key}-{value}" for key, value in parameters.items())
    return family + ("_" + values if values else "")


def fit_models() -> dict[str, Any]:
    freeze = assert_frozen()
    if freeze["externalOutcomesOpened"]:
        raise AssertionError("external outcomes were marked open before fitting")
    fit_rows, calibration_rows, extraction_checks = load_development()
    if len(fit_rows) != 880 or len(calibration_rows) != 220:
        raise AssertionError(
            f"development accounting changed: {len(fit_rows)}, {len(calibration_rows)}"
        )
    y_fit, y_cal = curve_matrix(fit_rows), curve_matrix(calibration_rows)
    static_fit, static_cal = (
        feature_matrix(fit_rows, False),
        feature_matrix(calibration_rows, False),
    )
    trans_fit, trans_cal = (
        feature_matrix(fit_rows, True),
        feature_matrix(calibration_rows, True),
    )
    pca = _pca_fit(y_fit)
    records: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}

    mean_model = {
        "family": "mean_shape",
        "mean_residual": np.mean(y_fit - y_fit[:, [0]], axis=0),
    }
    mean_pred = predict_model(mean_model, static_cal, y_cal[:, 0])
    name = "mean_shape"
    candidates[name] = mean_model
    records.append(
        {
            "candidate": name,
            "role": "baseline",
            "family": "mean_shape",
            **_condition_metrics(calibration_rows, y_cal, mean_pred),
        }
    )

    for alpha in RIDGE_ALPHAS:
        model = _fit_pca_ridge(static_fit, y_fit, pca, alpha)
        pred = predict_model(model, static_cal, y_cal[:, 0])
        name = _candidate_name("static_ridge", {"alpha": alpha})
        candidates[name] = model
        records.append(
            {
                "candidate": name,
                "role": "baseline",
                "family": "static_ridge",
                "alpha": alpha,
                **_condition_metrics(calibration_rows, y_cal, pred),
            }
        )

        model = _fit_pca_ridge(trans_fit, y_fit, pca, alpha)
        pred = predict_model(model, trans_cal, y_cal[:, 0])
        name = _candidate_name("transport_ridge", {"alpha": alpha})
        candidates[name] = model
        records.append(
            {
                "candidate": name,
                "role": "transport",
                "family": "transport_ridge",
                "alpha": alpha,
                **_condition_metrics(calibration_rows, y_cal, pred),
            }
        )

        model = _fit_markov(trans_fit, y_fit, alpha)
        pred = predict_model(model, trans_cal, y_cal[:, 0])
        name = _candidate_name("transport_markov", {"alpha": alpha})
        candidates[name] = model
        records.append(
            {
                "candidate": name,
                "role": "transport",
                "family": "transport_markov",
                "alpha": alpha,
                **_condition_metrics(calibration_rows, y_cal, pred),
            }
        )

    for depth, leaf in FOREST_CANDIDATES:
        model = _fit_pca_forest(trans_fit, y_fit, pca, depth, leaf)
        pred = predict_model(model, trans_cal, y_cal[:, 0])
        name = _candidate_name("transport_forest", {"depth": depth, "leaf": leaf})
        candidates[name] = model
        records.append(
            {
                "candidate": name,
                "role": "transport",
                "family": "transport_forest",
                "max_depth": depth,
                "min_samples_leaf": leaf,
                **_condition_metrics(calibration_rows, y_cal, pred),
            }
        )

    selection = (
        pd.DataFrame(records)
        .sort_values(["role", "pooled_rmse", "candidate"])
        .reset_index(drop=True)
    )
    transport = selection[selection.role == "transport"].copy()
    minimum = float(transport.pooled_rmse.min())
    near = transport[transport.pooled_rmse <= minimum + 0.0001].copy()
    order = {"transport_ridge": 0, "transport_markov": 1, "transport_forest": 2}
    near["simplicity"] = near.family.map(order)
    selected_row = near.sort_values(["simplicity", "pooled_rmse", "candidate"]).iloc[0]
    baseline_row = (
        selection[selection.role == "baseline"]
        .sort_values(["pooled_rmse", "candidate"])
        .iloc[0]
    )
    selected_name, baseline_name = (
        str(selected_row.candidate),
        str(baseline_row.candidate),
    )

    selected_cal_pred = predict_model(candidates[selected_name], trans_cal, y_cal[:, 0])
    calibration_by_condition: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(calibration_rows):
        calibration_by_condition[str(row["condition_id"])].append(index)
    absolute_residuals = []
    calibration_diagnostics = []
    for condition_id, indices in sorted(calibration_by_condition.items()):
        obs = y_cal[indices].mean(axis=0)
        pred = selected_cal_pred[indices].mean(axis=0)
        absolute_residuals.append(np.abs(obs - pred))
        calibration_diagnostics.append(
            {
                "condition_id": condition_id,
                "model": selected_name,
                "rmse": float(np.sqrt(np.mean((obs - pred) ** 2))),
                "mae": float(np.mean(np.abs(obs - pred))),
            }
        )
    conformal = np.quantile(np.stack(absolute_residuals), 0.90, axis=0, method="higher")

    all_rows = fit_rows + calibration_rows
    y_all = curve_matrix(all_rows)
    trans_all = feature_matrix(all_rows, True)
    static_all = feature_matrix(all_rows, False)
    selected_family = str(selected_row.family)
    if selected_family == "transport_ridge":
        final_primary = _fit_pca_ridge(
            trans_all, y_all, _pca_fit(y_all), float(selected_row.alpha)
        )
    elif selected_family == "transport_markov":
        final_primary = _fit_markov(trans_all, y_all, float(selected_row.alpha))
    else:
        depth_value = selected_row.max_depth
        depth = None if pd.isna(depth_value) else int(depth_value)
        final_primary = _fit_pca_forest(
            trans_all, y_all, _pca_fit(y_all), depth, int(selected_row.min_samples_leaf)
        )
    baseline_family = str(baseline_row.family)
    if baseline_family == "mean_shape":
        final_baseline = {
            "family": "mean_shape",
            "mean_residual": np.mean(y_all - y_all[:, [0]], axis=0),
        }
    else:
        final_baseline = _fit_pca_ridge(
            static_all, y_all, _pca_fit(y_all), float(baseline_row.alpha)
        )
    bundle = {
        "schema": "e04.s12.mechanism_model.v1",
        "research_step_id": "S12",
        "selected_name": selected_name,
        "selected_family": selected_family,
        "baseline_name": baseline_name,
        "static_feature_names": STATIC_NAMES,
        "transport_feature_names": TRANSPORT_NAMES,
        "primary": final_primary,
        "baseline": final_baseline,
        "conformal_q90": conformal,
    }
    model_dir = OUTPUT / "mechanism_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "policy_transport_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    write_parquet(selection, OUTPUT / "model_selection.parquet")
    write_parquet(
        pd.DataFrame(calibration_diagnostics),
        OUTPUT / "calibration_diagnostics.parquet",
    )
    feature_spec = {
        "schema": "e04.s12.feature_specification.v1",
        "static": STATIC_NAMES,
        "transport": TRANSPORT_NAMES,
        "selectedCandidate": selected_name,
        "selectedFamily": selected_family,
        "baselineCandidate": baseline_name,
        "pcaComponents": PCA_COMPONENTS,
        "externalOutcomesOpened": False,
    }
    write_json(model_dir / "feature_specification.json", feature_spec)
    extraction_summary = _validation_summary(extraction_checks)
    model_record = {
        "schema": "e04.s12.model_freeze.v1",
        "researchStepId": "S12",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "externalOutcomesOpened": False,
        "developmentFitRows": len(fit_rows),
        "developmentCalibrationRows": len(calibration_rows),
        "developmentConditions": len({row["condition_id"] for row in all_rows}),
        "selectedCandidate": selected_name,
        "selectedFamily": selected_family,
        "baselineCandidate": baseline_name,
        "modelSha256": sha256_file(model_path),
        "selectionTableSha256": sha256_file(OUTPUT / "model_selection.parquet"),
        "calibrationExtractionValidation": extraction_summary,
    }
    write_json(OUTPUT / "model_freeze_record.json", model_record)
    return model_record


def _validation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "passed": False}
    return {
        "rows": len(rows),
        "occupancyPermutationFailures": sum(
            not bool(row["occupancy_permutation_passed"]) for row in rows
        ),
        "labelCountFailures": sum(not bool(row["label_count_passed"]) for row in rows),
        "curveMaxError": max(float(row["curve_max_error"]) for row in rows),
        "structuralIdentityMaxError": max(
            float(row["structural_identity_max_error"]) for row in rows
        ),
        "activationAccountingMaxError": max(
            float(row["activation_accounting_max_error"]) for row in rows
        ),
        "fluxConservationMaxError": max(
            float(row["flux_conservation_error"]) for row in rows
        ),
        "residenceBoundFailures": sum(
            not bool(row["residence_bounds_passed"]) for row in rows
        ),
        "passed": all(
            bool(row["occupancy_permutation_passed"])
            and bool(row["label_count_passed"])
            and float(row["curve_max_error"]) <= 1e-12
            and float(row["structural_identity_max_error"]) <= 1e-12
            and float(row["activation_accounting_max_error"]) <= 1e-12
            and float(row["flux_conservation_error"]) <= 1e-12
            and bool(row["residence_bounds_passed"])
            for row in rows
        ),
    }


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) < 1e-14 or np.std(values[1:]) < 1e-14:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _curve_stats(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "peak_observed": float(np.max(observed)),
        "peak_predicted": float(np.max(predicted)),
        "peak_absolute_error": abs(float(np.max(observed) - np.max(predicted))),
        "positive_area_observed": float(np.trapezoid(np.maximum(observed, 0.0), GRID)),
        "positive_area_predicted": float(
            np.trapezoid(np.maximum(predicted, 0.0), GRID)
        ),
        "positive_area_absolute_error": abs(
            float(
                np.trapezoid(np.maximum(observed, 0.0), GRID)
                - np.trapezoid(np.maximum(predicted, 0.0), GRID)
            )
        ),
        "final_absolute_error": abs(float(observed[-1] - predicted[-1])),
        "mean_residual": float(np.mean(residual)),
        "lag1_residual_acf": _lag1(residual),
    }


def evaluate_model() -> dict[str, Any]:
    assert_frozen()
    model_record = json.loads((OUTPUT / "model_freeze_record.json").read_text())
    model_path = OUTPUT / "mechanism_model/policy_transport_model.joblib"
    if (
        model_record["modelSha256"] != sha256_file(model_path)
        or model_record["externalOutcomesOpened"]
    ):
        raise AssertionError("model was not sealed before evaluation")
    bundle = joblib.load(model_path)
    covariates = _scenario_covariates()
    all_features: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    regime_counts = {}
    for regime in REGIMES:
        count = 0
        for raw in _read_jsonl(S07_CACHE / f"{regime}.jsonl"):
            feature, ledger, check = extract_run(
                raw, covariates[str(raw["scenario_id"])]
            )
            all_features.append(feature)
            transport_rows.extend(ledger)
            validation_rows.append(check)
            split_rows.append(
                {
                    "scenario_id": feature["scenario_id"],
                    "condition_id": feature["condition_id"],
                    "replicate_ordinal": feature["replicate_ordinal"],
                    "regime": feature["regime"],
                    "split": feature["split"],
                    "outcome_opened_after_model_freeze": feature["split"].startswith(
                        "native_"
                    )
                    or feature["split"].startswith("intervention_"),
                }
            )
            count += 1
        regime_counts[regime] = count
    if len(all_features) != EXPECTED_ROWS or any(
        value != EXPECTED_SCENARIOS for value in regime_counts.values()
    ):
        raise AssertionError("evaluation run accounting changed")
    split_frame = pd.DataFrame(split_rows)
    split_counts = (
        split_frame.groupby(["split", "regime"]).size().rename("rows").reset_index()
    )
    if set(
        split_frame[split_frame.split == "development_fit"].replicate_ordinal
    ) != set(FIT_REPLICATES):
        raise AssertionError("fit replicate assignment changed")
    if set(
        split_frame[split_frame.split == "development_calibration"].replicate_ordinal
    ) != set(CALIBRATION_REPLICATES):
        raise AssertionError("calibration replicate assignment changed")

    evaluation_rows = [
        row for row in all_features if not row["split"].startswith("development")
    ]
    observed = curve_matrix(evaluation_rows)
    x_transport = feature_matrix(evaluation_rows, True)
    x_static = feature_matrix(evaluation_rows, False)
    y0 = observed[:, 0]
    primary = predict_model(bundle["primary"], x_transport, y0)
    baseline = predict_model(bundle["baseline"], x_static, y0)
    conformal = np.asarray(bundle["conformal_q90"], dtype=float)

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(evaluation_rows):
        by_key[(str(row["regime"]), str(row["condition_id"]))].append(index)
    prediction_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for (regime, condition_id), indices in sorted(by_key.items()):
        first = evaluation_rows[indices[0]]
        obs = observed[indices].mean(axis=0)
        pred = primary[indices].mean(axis=0)
        base = baseline[indices].mean(axis=0)
        category = str(first["split"])
        primary_stats = _curve_stats(obs, pred)
        baseline_stats = _curve_stats(obs, base)
        coverage = float(np.mean((obs >= pred - conformal) & (obs <= pred + conformal)))
        performance_rows.append(
            {
                "regime": regime,
                "condition_id": condition_id,
                "evaluation_stratum": category,
                "input_profile": first["input_profile"],
                "policy_set_label": first["policy_set_label"],
                "composition_profile": first["composition_profile"],
                "correlation_profile": first["correlation_profile"],
                "model": bundle["selected_name"],
                **primary_stats,
                "conformal_coverage": coverage,
                "baseline_model": bundle["baseline_name"],
                **{f"baseline_{key}": value for key, value in baseline_stats.items()},
                "rmse_improvement": 1.0 - primary_stats["rmse"] / baseline_stats["rmse"]
                if baseline_stats["rmse"]
                else 0.0,
            }
        )
        for index, progress in enumerate(GRID):
            prediction_rows.append(
                {
                    "regime": regime,
                    "condition_id": condition_id,
                    "evaluation_stratum": category,
                    "input_profile": first["input_profile"],
                    "policy_set_label": first["policy_set_label"],
                    "composition_profile": first["composition_profile"],
                    "correlation_profile": first["correlation_profile"],
                    "grid_index": index,
                    "progress": progress,
                    "observed": obs[index],
                    "predicted": pred[index],
                    "baseline_predicted": base[index],
                    "prediction_lower90": pred[index] - conformal[index],
                    "prediction_upper90": pred[index] + conformal[index],
                    "residual": obs[index] - pred[index],
                }
            )
    performance = pd.DataFrame(performance_rows)
    predictions = pd.DataFrame(prediction_rows)
    group_rows = []
    for keys, group in performance.groupby(["evaluation_stratum", "regime"], sort=True):
        group_rows.append(
            {
                "evaluation_stratum": keys[0],
                "regime": keys[1],
                "conditions": len(group),
                "rmse": float(np.sqrt(np.mean(group.rmse**2))),
                "baseline_rmse": float(np.sqrt(np.mean(group.baseline_rmse**2))),
                "rmse_improvement": 1.0
                - float(np.sqrt(np.mean(group.rmse**2)))
                / float(np.sqrt(np.mean(group.baseline_rmse**2))),
                "peak_mae": float(group.peak_absolute_error.mean()),
                "positive_area_mae": float(group.positive_area_absolute_error.mean()),
                "absolute_mean_residual": abs(float(group.mean_residual.mean())),
                "median_lag1_residual_acf": float(group.lag1_residual_acf.median()),
                "coverage": float(group.conformal_coverage.mean()),
            }
        )
    grouped = pd.DataFrame(group_rows)
    native = performance[
        performance.evaluation_stratum.isin(
            ["native_composition_holdout", "native_association_holdout"]
        )
    ]
    native_summary = {
        "conditions": len(native),
        "rmse": float(np.sqrt(np.mean(native.rmse**2))),
        "baseline_rmse": float(np.sqrt(np.mean(native.baseline_rmse**2))),
        "rmse_improvement": 1.0
        - float(np.sqrt(np.mean(native.rmse**2)))
        / float(np.sqrt(np.mean(native.baseline_rmse**2))),
        "peak_mae": float(native.peak_absolute_error.mean()),
        "positive_area_mae": float(native.positive_area_absolute_error.mean()),
        "absolute_mean_residual": abs(float(native.mean_residual.mean())),
        "coverage": float(native.conformal_coverage.mean()),
    }
    activation_group = performance[performance.regime == "activation_equal"]
    activation_summary = {
        "conditions": len(activation_group),
        "rmse": float(np.sqrt(np.mean(activation_group.rmse**2))),
        "baseline_rmse": float(np.sqrt(np.mean(activation_group.baseline_rmse**2))),
        "rmse_improvement": 1.0
        - float(np.sqrt(np.mean(activation_group.rmse**2)))
        / float(np.sqrt(np.mean(activation_group.baseline_rmse**2))),
        "peak_mae": float(activation_group.peak_absolute_error.mean()),
        "positive_area_mae": float(
            activation_group.positive_area_absolute_error.mean()
        ),
    }
    native_gates = {
        "rmse": native_summary["rmse"] <= 0.05,
        "improvement": native_summary["rmse_improvement"] >= 0.25,
        "peak": native_summary["peak_mae"] <= 0.05,
        "positive_area": native_summary["positive_area_mae"] <= 0.03,
        "bias": native_summary["absolute_mean_residual"] <= 0.02,
        "coverage": 0.80 <= native_summary["coverage"] <= 0.98,
    }
    activation_gates = {
        "rmse": activation_summary["rmse"] <= 0.06,
        "improvement": activation_summary["rmse_improvement"] >= 0.10,
        "peak": activation_summary["peak_mae"] <= 0.06,
        "positive_area": activation_summary["positive_area_mae"] <= 0.04,
    }
    absolute_pass = all(
        value for key, value in native_gates.items() if key != "improvement"
    ) and all(value for key, value in activation_gates.items() if key != "improvement")
    if all(native_gates.values()) and all(activation_gates.values()):
        outcome = "supportive"
    elif absolute_pass:
        outcome = "null"
    else:
        outcome = "constraining/contradictory"
    decision = {
        "schema": "e04.s12.adequacy_decision.v1",
        "researchStepId": "S12",
        "selectedModel": bundle["selected_name"],
        "strongestBaseline": bundle["baseline_name"],
        "nativeExternal": native_summary,
        "nativeGates": native_gates,
        "activationEqual": activation_summary,
        "activationEqualGates": activation_gates,
        "outcomeClassification": outcome,
        "predictivelyAdequate": all(native_gates.values())
        and all(activation_gates.values()),
        "causalIdentification": False,
    }
    failure_rows = []
    dimensions = [
        "input_profile",
        "policy_set_label",
        "composition_profile",
        "correlation_profile",
        "regime",
        "condition_id",
    ]
    for dimension in dimensions:
        for value, group in performance.groupby(dimension, dropna=False, sort=True):
            rmse = float(np.sqrt(np.mean(group.rmse**2)))
            baseline_rmse = float(np.sqrt(np.mean(group.baseline_rmse**2)))
            limit = 0.06 if bool((group.regime == "activation_equal").all()) else 0.05
            failure_rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "conditions": len(group),
                    "rmse": rmse,
                    "baseline_rmse": baseline_rmse,
                    "rmse_improvement": 1.0 - rmse / baseline_rmse
                    if baseline_rmse
                    else 0.0,
                    "mean_residual": float(group.mean_residual.mean()),
                    "median_lag1_residual_acf": float(group.lag1_residual_acf.median()),
                    "flagged": rmse > limit or rmse > 1.10 * baseline_rmse,
                }
            )
    failures = pd.DataFrame(failure_rows)
    validation = _validation_summary(validation_rows)
    if not validation["passed"]:
        raise AssertionError(f"transport validation failed: {validation}")
    transport = pd.DataFrame(transport_rows)
    transport_summary = transport.groupby(["regime", "policy"], sort=True)[
        ["actor_velocity", "collision_load", "experienced_flux", "residence_time"]
    ].agg(["mean", "median", "std", "min", "max"])
    transport_summary.columns = [
        "_".join(column) for column in transport_summary.columns
    ]
    transport_summary = transport_summary.reset_index()
    write_parquet(transport, OUTPUT / "transport_statistics.parquet")
    transport_summary.to_csv(OUTPUT / "transport_summary.csv", index=False)
    write_parquet(split_frame, OUTPUT / "split_manifest.parquet")
    write_parquet(split_counts, OUTPUT / "split_accounting.parquet")
    write_parquet(predictions, OUTPUT / "heldout_predictions.parquet")
    write_parquet(performance, OUTPUT / "heldout_performance.parquet")
    write_parquet(grouped, OUTPUT / "fit_diagnostics.parquet")
    write_parquet(failures, OUTPUT / "failure_regions.parquet")
    residuals = predictions[
        [
            "regime",
            "condition_id",
            "evaluation_stratum",
            "grid_index",
            "progress",
            "residual",
        ]
    ].copy()
    write_parquet(residuals, OUTPUT / "residual_diagnostics.parquet")
    write_json(OUTPUT / "adequacy_decision.json", decision)
    write_json(
        OUTPUT / "transport_validation.json",
        {
            "schema": "e04.s12.transport_validation.v1",
            "researchStepId": "S12",
            **validation,
            "regimeCounts": regime_counts,
        },
    )
    write_json(
        OUTPUT / "evaluation_accounting.json",
        {
            "schema": "e04.s12.evaluation_accounting.v1",
            "researchStepId": "S12",
            "sourceRows": len(all_features),
            "evaluationScenarioRows": len(evaluation_rows),
            "transportRows": len(transport),
            "predictionRows": len(predictions),
            "performanceRows": len(performance),
            "regimeCounts": regime_counts,
            "splitCounts": split_counts.to_dict(orient="records"),
        },
    )
    opened = {
        **model_record,
        "externalOutcomesOpened": True,
        "externalOutcomesOpenedAt": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUTPUT / "model_freeze_record.json", opened)
    _write_model_diagnostics(bundle)
    _plots(transport, grouped, performance, predictions)
    return decision


def _write_model_diagnostics(bundle: Mapping[str, Any]) -> None:
    primary = bundle["primary"]
    names = STATIC_NAMES + TRANSPORT_NAMES
    rows = []
    if primary["family"] == "pca_forest":
        for name, value in zip(
            names, primary["regressor"].feature_importances_, strict=True
        ):
            rows.append(
                {
                    "feature": name,
                    "diagnostic": "impurity_importance",
                    "value": float(value),
                }
            )
    elif primary["family"] == "pca_ridge":
        importance = np.linalg.norm(primary["regressor"].coef_, axis=0)
        for name, value in zip(names, importance, strict=True):
            rows.append(
                {
                    "feature": name,
                    "diagnostic": "coefficient_l2_across_pca_scores",
                    "value": float(value),
                }
            )
    else:
        coefficients = np.asarray(primary["regressor"].coef_)
        markov_names = names + [
            "current_aggregation",
            "progress",
            "progress_squared",
            "sin_progress",
            "cos_progress",
        ]
        for name, value in zip(markov_names, coefficients, strict=True):
            rows.append(
                {
                    "feature": name,
                    "diagnostic": "standardized_markov_coefficient",
                    "value": float(value),
                }
            )
    write_parquet(
        pd.DataFrame(rows), OUTPUT / "mechanism_model/model_diagnostics.parquet"
    )


def _save_figure(fig: Any, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _plots(
    transport: pd.DataFrame,
    grouped: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    native = transport[transport.regime == "native_control"]
    metrics = ["actor_velocity", "collision_load", "experienced_flux", "residence_time"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric in zip(axes.flat, metrics, strict=True):
        data = [
            native.loc[native.policy == policy, metric].to_numpy()
            for policy in POLICIES
        ]
        axis.boxplot(data, tick_labels=POLICIES, showfliers=False)
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("Native policy-specific transport distributions")
    fig.tight_layout()
    _save_figure(fig, "policy_transport_distributions")

    plot_group = grouped.copy()
    labels = [
        f"{row.evaluation_stratum}\n{row.regime}" for row in plot_group.itertuples()
    ]
    x = np.arange(len(plot_group))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(x - 0.2, plot_group.rmse, 0.4, label="transport model")
    axis.bar(x + 0.2, plot_group.baseline_rmse, 0.4, label="strongest baseline")
    axis.axhline(
        0.05, color="black", linestyle="--", linewidth=1, label="native RMSE gate"
    )
    axis.set_xticks(x, labels, rotation=40, ha="right")
    axis.set_ylabel("Condition-level trajectory RMSE")
    axis.legend()
    axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    _save_figure(fig, "heldout_model_performance")

    selected = (
        performance.sort_values("rmse").groupby("evaluation_stratum", sort=True).head(1)
    )
    selected = (
        pd.concat(
            [
                selected,
                performance.sort_values("rmse", ascending=False)
                .groupby("evaluation_stratum", sort=True)
                .head(1),
            ]
        )
        .drop_duplicates(["regime", "condition_id"])
        .head(8)
    )
    fig, axes = plt.subplots(
        math.ceil(len(selected) / 2),
        2,
        figsize=(12, 3.2 * math.ceil(len(selected) / 2)),
        squeeze=False,
    )
    for axis, item in zip(axes.flat, selected.itertuples(), strict=False):
        group = predictions[
            (predictions.regime == item.regime)
            & (predictions.condition_id == item.condition_id)
        ]
        axis.plot(group.progress, group.observed, color="black", label="observed")
        axis.plot(group.progress, group.predicted, color="#1f77b4", label="transport")
        axis.plot(
            group.progress,
            group.baseline_predicted,
            color="#ff7f0e",
            linestyle="--",
            label="baseline",
        )
        axis.fill_between(
            group.progress,
            group.prediction_lower90,
            group.prediction_upper90,
            color="#1f77b4",
            alpha=0.15,
        )
        axis.set_title(f"{item.regime}: {item.condition_id}\nRMSE={item.rmse:.3f}")
        axis.set_xlabel("accepted-swap progress")
        axis.set_ylabel("corrected aggregation")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, "heldout_trajectory_predictions")

    worst = performance.sort_values("rmse", ascending=False).head(25).copy()
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        np.arange(len(worst)),
        worst.rmse.iloc[::-1],
        color=np.where(
            worst.rmse.iloc[::-1] > worst.baseline_rmse.iloc[::-1], "#d62728", "#1f77b4"
        ),
    )
    axis.set_yticks(
        np.arange(len(worst)),
        [f"{r.regime}: {r.condition_id}" for r in worst.iloc[::-1].itertuples()],
        fontsize=7,
    )
    axis.set_xlabel("Trajectory RMSE")
    axis.axvline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_title("Largest held-out residual regions (red: worse than baseline)")
    axis.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    _save_figure(fig, "failure_regions")


def finalize() -> dict[str, Any]:
    assert_frozen()
    decision = json.loads((OUTPUT / "adequacy_decision.json").read_text())
    validation = json.loads((OUTPUT / "transport_validation.json").read_text())
    accounting = json.loads((OUTPUT / "evaluation_accounting.json").read_text())
    upstream_after = {}
    for step in [f"S{i:02d}" for i in range(1, 12)] + ["S11R"]:
        upstream_after[step] = _verify_manifest(
            Path("/artifacts/research_steps") / step
        )
    immutability = {
        "schema": "e04.s12.upstream_immutability.v1",
        "researchStepId": "S12",
        "steps": upstream_after,
        "allPassed": all(item["passed"] for item in upstream_after.values()),
        "s13Absent": not S13_DIR.exists(),
    }
    write_json(OUTPUT / "upstream_immutability_audit.json", immutability)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikitLearn": __import__("sklearn").__version__,
        "workers": 8,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "gitCommitBeforeS12": _git("rev-parse", "HEAD"),
    }
    write_json(OUTPUT / "environment.json", environment)
    validation_summary = {
        "schema": "e04.s12.validation_summary.v1",
        "researchStepId": "S12",
        "success": bool(
            validation["passed"]
            and immutability["allPassed"]
            and immutability["s13Absent"]
        ),
        "transport": validation,
        "accounting": accounting,
        "upstreamImmutability": immutability["allPassed"],
        "s13Absent": immutability["s13Absent"],
        "modelFreezePassed": True,
    }
    write_json(OUTPUT / "validation_summary.json", validation_summary)
    status = {
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": validation_summary["success"],
        "status": "complete" if validation_summary["success"] else "blocked",
        "artifactsWritten": [],
        "validationResult": "passed" if validation_summary["success"] else "failed",
        "outcomeClassification": decision["outcomeClassification"],
        "caveatsOrBlockers": [
            "Complete-run summaries support retrospective prediction, not online forecasting or causal identification.",
            "S11R intervention infeasibility remains unresolved and S11 Shapley estimates were excluded.",
            "Collision load is an operational blocked-action/no-op measure; residence is interval-censored on the 1% progress grid.",
        ],
        "recommendedNextAction": "Return S12 to the Chief Scientist; do not start S13 automatically.",
    }
    write_json(OUTPUT / "status.json", status)
    return {"decision": decision, "validation": validation_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "fit", "evaluate", "finalize"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "freeze":
        result = freeze_design()
    elif args.stage == "fit":
        result = fit_models()
    elif args.stage == "evaluate":
        result = evaluate_model()
    else:
        result = finalize()
    print(json.dumps(_native(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
