"""Frozen S10 native-event discovery and independent reproduction.

This module consumes the byte-frozen S10P population, feature registry, and
method registry.  It never reads validation/confirmation outcomes, rejected
surrogate artifacts, S07 arm signals, or historical quarantine rows.  Spatial
episodes are evaluated through the authoritative E06 DSL adapter so the
authentic ordered transition summaries are available; only audited counts and
state-hash change indicators leave that boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.mixture import GaussianMixture

from src.environment_suite import portfolio_action
from src.environment_suite.contracts import ScenarioRecord, SuiteValidationError
from src.environment_suite.dsl_adapters import run_spatial_dsl_episode
from src.environment_suite.runners import _morph_context
from src.environment_suite.suite import EnvironmentSuite
from src.environment_suite import Split
from src.phenotype_discovery.native_features import (
    canonical_sha256,
    exact_change_points,
    extract_native_event_features,
    validate_ordered_transition_summaries,
)
from src.policy_dsl import compile_policy


ARTIFACT_ROOT = Path("/artifacts/research_steps")
S02 = ARTIFACT_ROOT / "S02"
S05 = ARTIFACT_ROOT / "S05"
S09 = ARTIFACT_ROOT / "S09"
S10P = ARTIFACT_ROOT / "S10P"
REPOSITORY = Path(__file__).resolve().parents[2]
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
SPATIAL_TASKS = (
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)
K_GRID = (2, 3, 4, 5, 6)
BOOTSTRAP_REPLICATES = 200
NULL_REPLICATES = 500
ANOMALY_SEED = 71020260723

_WORKER_CONTEXT: Any = None
_WORKER_TASKS: dict[str, Any] = {}
_WORKER_BASES: dict[str, ScenarioRecord] = {}
_WORKER_BUNDLES: dict[str, dict[str, Any]] = {}
_WORKER_SCENARIOS: dict[tuple[str, str, int], dict[str, Any]] = {}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_seed(*parts: Any) -> int:
    payload = "/".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fold_for_family(scenario_family_id: str) -> int:
    return int(hashlib.sha256(scenario_family_id.encode("utf-8")).hexdigest(), 16) % 5


def status_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["stopReason"]),
            f"failed={str(bool(row['failed'])).lower()}",
            f"censored={str(bool(row['censored'])).lower()}",
        )
    )


def _base_train_records() -> tuple[dict[str, Any], dict[str, ScenarioRecord]]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    bases = {
        record.task_id: record
        for record in suite.records.values()
        if record.task_id in SPATIAL_TASKS and record.split is Split.TRAIN
    }
    if set(bases) != set(SPATIAL_TASKS):
        raise RuntimeError("S10 spatial training bases are incomplete")
    return dict(suite.tasks), bases


def _load_candidate_bundles() -> dict[str, dict[str, Any]]:
    candidates = read_jsonl(S10P / "candidate_population.jsonl")
    catalog = {
        str(row["policySha256"]): row["document"]
        for row in read_jsonl(S05 / "policy_catalog.jsonl")
    }
    compressed = {
        str(row["variantConfigurationId"]): row
        for row in read_jsonl(S09 / "compressed_policies.jsonl")
    }
    bundles: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidateId"])
        configuration = deepcopy(candidate["structuralConfiguration"])
        if candidate["candidateRole"] == "s09_compressed":
            source = compressed.get(candidate_id)
            if source is None:
                raise RuntimeError("compressed candidate lacks frozen S09 bundle")
            documents = deepcopy(source["documentsByPolicySha256"])
        else:
            documents = {}
            for member in configuration["members"]:
                policy_sha = str(member["policySha256"])
                if policy_sha not in catalog:
                    raise RuntimeError("parent member lacks authoritative S05 document")
                documents[policy_sha] = deepcopy(catalog[policy_sha])
        expected_hashes = {
            str(item["policySha256"]) for item in configuration["members"]
        }
        if set(documents) != expected_hashes:
            raise RuntimeError("candidate member/document identity mismatch")
        for policy_sha, document in documents.items():
            if compile_policy(document).policy_sha256 != policy_sha:
                raise RuntimeError("candidate policy failed canonical recompilation")
        bundles[candidate_id] = {
            "candidate": candidate,
            "configuration": configuration,
            "documentsByPolicySha256": documents,
        }
    if len(bundles) != 14:
        raise RuntimeError("S10 must resolve exactly 14 candidate bundles")
    return bundles


def _load_scenarios() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = read_jsonl(S10P / "scenario_population.jsonl")
    mapping = {
        (str(row["phase"]), str(row["taskId"]), int(row["scenarioFamilyOrdinal"])): row
        for row in rows
    }
    if len(rows) != 768 or len(mapping) != 768:
        raise RuntimeError("S10 frozen scenario population changed")
    return mapping


def load_roster(phase: str | None = None) -> list[dict[str, Any]]:
    rows = pq.read_table(S10P / "s10_logical_roster.parquet").to_pylist()
    if phase is not None:
        rows = [row for row in rows if row["phase"] == phase]
    return rows


def initialize_worker() -> None:
    global _WORKER_CONTEXT, _WORKER_TASKS, _WORKER_BASES
    global _WORKER_BUNDLES, _WORKER_SCENARIOS
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    _WORKER_CONTEXT = _morph_context()
    _WORKER_TASKS, _WORKER_BASES = _base_train_records()
    _WORKER_BUNDLES = _load_candidate_bundles()
    _WORKER_SCENARIOS = _load_scenarios()


def _action_for(bundle: Mapping[str, Any], task_id: str):
    configuration = deepcopy(bundle["configuration"])
    configuration["taskId"] = task_id
    documents = bundle["documentsByPolicySha256"]
    ordered = [
        documents[str(item["policySha256"])] for item in configuration["members"]
    ]
    return portfolio_action(ordered, configuration), configuration


def _episode_definition(base: ScenarioRecord, scenario: Mapping[str, Any]):
    native_scenario_id = str(base.public_parameters["nativeScenarioId"])
    definition = next(
        item
        for item in _WORKER_CONTEXT.episodes
        if item.scenario_id == native_scenario_id
    )
    from dataclasses import replace

    return replace(definition, scenario_id=str(scenario["counterScheduleKey"]))


def _series_projection(summaries: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {
        "proposalCount": [int(row["proposalCount"]) for row in summaries],
        "acceptedCount": [int(row["acceptedCount"]) for row in summaries],
        "conflictLosses": [int(row["conflictLosses"]) for row in summaries],
        "invalidProposals": [int(row["invalidProposals"]) for row in summaries],
        "stateHashChanged": [
            str(row["preStateSha256"]) != str(row["postStateSha256"])
            for row in summaries
        ],
    }


def evaluate_reservation(reservation: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one frozen logical reservation with exact double execution."""

    if reservation.get("split") != "train":
        raise SuiteValidationError("S10 reservation left the training split")
    task_id = str(reservation["taskId"])
    if task_id not in SPATIAL_TASKS:
        raise SuiteValidationError("summary-only task entered substantive S10")
    key = (
        str(reservation["phase"]),
        task_id,
        int(reservation["scenarioFamilyOrdinal"]),
    )
    scenario = _WORKER_SCENARIOS[key]
    if scenario["scenarioCommitmentSha256"] != reservation["scenarioCommitmentSha256"]:
        raise RuntimeError("scenario commitment mismatch")
    bundle = _WORKER_BUNDLES[str(reservation["candidateId"])]
    candidate = bundle["candidate"]
    if (
        candidate["candidateDefinitionSha256"]
        != reservation["candidateDefinitionSha256"]
    ):
        raise RuntimeError("candidate definition commitment mismatch")
    action, configuration = _action_for(bundle, task_id)
    base = _WORKER_BASES[task_id]
    definition = _episode_definition(base, scenario)
    if definition.transitions != 32 or definition.actor_batch_size != 4:
        raise RuntimeError("native E06 horizon changed")
    target_id = str(base.public_parameters["targetId"])
    started = time.perf_counter()
    first = run_spatial_dsl_episode(
        _WORKER_CONTEXT, definition, action, target_id=target_id
    )
    second = run_spatial_dsl_episode(
        _WORKER_CONTEXT, definition, action, target_id=target_id
    )
    elapsed = time.perf_counter() - started
    replay_pass = first == second
    summary_validation = validate_ordered_transition_summaries(
        first["transitionSummaries"], expected_count=32
    )
    authority = {key: bool(value) for key, value in first["authorityAudit"].items()}
    validation = {
        "exactReplay": replay_pass,
        "fixedTransitionCount": len(first["transitionSummaries"]) == 32,
        "actorBatchSize": int(first["actorBatchSize"]) == 4,
        "stateHashChain": bool(summary_validation["stateHashChainPass"]),
        "configurationChargedOnce": int(first["e06ChannelLedger"]["configurationBits"])
        == 0,
        "portfolioAssignmentAudit": "portfolioAssignmentAudit" in first,
        "portfolioDispatchCount": sum(
            int(value)
            for value in first["portfolioAssignmentAudit"][
                "assignmentCountsByPolicySha256"
            ].values()
        )
        == 128,
        **authority,
    }
    false_validation = sorted(key for key, value in validation.items() if not value)
    failed = bool(false_validation)
    censored = not bool(first["offlineEvaluation"]["conjunctiveCompletionByBudget"])
    environment = _WORKER_CONTEXT.environments[definition.environment_id]
    payload = {
        "event": {
            "schemaVersion": str(first["schemaVersion"]),
            "episodeSha256": str(first["episodeSha256"]),
            "transitionCount": 32,
            "actorBatchSize": 4,
            "onlineGlobalCompletionComputed": False,
        },
        "costs": {"e06MovementLedger": deepcopy(first["movementLedger"])},
        "status": {
            "stopReason": str(first["stopReason"]),
            "failed": failed,
            "censored": censored,
        },
        "horizon": {
            "nativeUnit": _WORKER_TASKS[task_id].horizon.native_unit,
            "budget": 32,
            "actorSlotsPerTransition": 4,
        },
        "structuralScenarioSize": len(environment.occupiable_sites),
        "traceSelectionReason": "all_event_summaries",
        "orderedEventSummaries": first["transitionSummaries"],
    }
    feature_record = extract_native_event_features(task_id, payload)
    if len(feature_record["analysisFeatures"]) != 18:
        raise RuntimeError("spatial S10 row did not expose 18 frozen features")
    if any(
        value["state"] != "observed_or_exact_native_derived"
        for value in feature_record["availability"].values()
    ):
        raise RuntimeError("spatial S10 feature unexpectedly unavailable")
    series = _series_projection(first["transitionSummaries"])
    cost_ledgers = {
        "e06MovementLedger": deepcopy(first["movementLedger"]),
        "e06ObservationLedger": deepcopy(first["observationLedger"]),
        "e06ChannelLedger": deepcopy(first["e06ChannelLedger"]),
        "dslRuntimeLedger": deepcopy(first["dslRuntimeLedger"]),
        "dslCommunicationLedger": deepcopy(first["dslCommunicationLedger"]),
        "licensedCapabilityLedger": deepcopy(first["licensedCapabilityLedger"]),
        "portfolioStructuralLedger": deepcopy(first["portfolioStructuralLedger"]),
        "portfolioCoordinationLedger": deepcopy(first["portfolioCoordinationLedger"]),
    }
    body = {
        "schemaVersion": "e07.s10.native-event-evaluation-row.v1",
        "researchStepId": "S10",
        "logicalOrdinal": int(reservation["logicalOrdinal"]),
        "logicalReservationId": str(reservation["logicalReservationId"]),
        "phase": str(reservation["phase"]),
        "taskId": task_id,
        "split": "train",
        "scenarioFamilyOrdinal": int(reservation["scenarioFamilyOrdinal"]),
        "scenarioFamilyId": str(reservation["scenarioFamilyId"]),
        "scenarioCommitmentSha256": str(reservation["scenarioCommitmentSha256"]),
        "candidateId": str(reservation["candidateId"]),
        "candidateRole": str(reservation["candidateRole"]),
        "candidateDefinitionSha256": str(reservation["candidateDefinitionSha256"]),
        "actionSha256": str(action.policy_sha256),
        "configurationMode": str(configuration["mode"]),
        "memberPolicySha256": [
            str(item["policySha256"]) for item in configuration["members"]
        ],
        "stopReason": str(first["stopReason"]),
        "failed": failed,
        "censored": censored,
        "statusStratum": status_key(
            {
                "stopReason": first["stopReason"],
                "failed": failed,
                "censored": censored,
            }
        ),
        "replayPass": replay_pass,
        "nativeContractPass": not false_validation,
        "nativeContractErrors": false_validation,
        "featureRecordSha256": feature_record["featureRecordSha256"],
        "analysisFeatures": feature_record["analysisFeatures"],
        "confounds": feature_record["confounds"],
        "availability": feature_record["availability"],
        "orderedSummaryCommitmentSha256": summary_validation["contentSha256"],
        "orderedEventSeries": series,
        "nativeEventSha256": canonical_sha256(
            "E07/S10/native-event-projection/v1", payload["event"]
        ),
        "nativeCostLedgers": cost_ledgers,
        "nativeCostSha256": canonical_sha256(
            "E07/S10/native-cost-ledgers/v1", cost_ledgers
        ),
        "physicalExecutions": 2,
        "outcomePlaneReadForFeatures": False,
        "statusDerivedByNativeRunnerSemantics": True,
        "completeTrajectoryReconstructed": False,
        "claimBoundary": _WORKER_TASKS[task_id].claim_boundary,
        "elapsedSeconds": elapsed,
        "workerPid": os.getpid(),
    }
    body["logicalResultSha256"] = canonical_sha256(
        "E07/S10/logical-result/v1",
        {
            key: value
            for key, value in body.items()
            if key not in {"elapsedSeconds", "workerPid"}
        },
    )
    return body


def execute_phase(
    reservations: Sequence[Mapping[str, Any]], *, workers: int = 8
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    if len({row["logicalReservationId"] for row in reservations}) != len(reservations):
        raise RuntimeError("duplicate logical reservation")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker
    ) as executor:
        futures = {
            executor.submit(evaluate_reservation, row): index
            for index, row in enumerate(reservations)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(
                    {
                        "position": index,
                        "logicalReservationId": reservations[index][
                            "logicalReservationId"
                        ],
                        "errorType": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "errorMessage": str(exc),
                    }
                )
    if failures:
        raise RuntimeError(
            "S10 fail-atomic phase execution failed: "
            + json.dumps(failures[:5], sort_keys=True)
        )
    results.sort(key=lambda row: int(row["logicalOrdinal"]))
    return results, {
        "logicalRows": len(reservations),
        "physicalExecutions": 2 * len(reservations),
        "publishedRows": len(results),
        "failedWorkerRows": 0,
        "workers": workers,
        "numericThreadsPerWorker": 1,
        "failAtomic": True,
        "wallSeconds": time.perf_counter() - started,
    }


def rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["logicalOrdinal"])):
        digest.update(str(row["logicalResultSha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = sorted({name for row in rows for name in row["analysisFeatures"]})
    return names


def _covariates(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    status_values = sorted({str(row["statusStratum"]) for row in rows})
    result = []
    for row in rows:
        confounds = row["confounds"]
        horizon = confounds["nativeHorizon"]
        budget = float(horizon.get("budget", 32))
        length = float(confounds["observedEventLength"])
        result.append(
            [
                math.log1p(length),
                length / max(1.0, budget),
                float(confounds["structuralScenarioSize"]),
                *[float(str(row["statusStratum"]) == value) for value in status_values],
            ]
        )
    return np.asarray(result, dtype=float)


def fit_preprocessing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit frozen availability, cross-fitted ridge, and MAD transforms."""

    if not rows:
        raise RuntimeError("cannot fit preprocessing without discovery rows")
    feature_names = _feature_names(rows)
    support = {
        feature: sum(feature in row["analysisFeatures"] for row in rows) / len(rows)
        for feature in feature_names
    }
    eligible = sorted(feature for feature, value in support.items() if value >= 0.90)
    if len(eligible) < 5:
        raise RuntimeError("fewer than five frozen features have 90% support")
    folds = np.asarray(
        [fold_for_family(str(row["scenarioFamilyId"])) for row in rows], dtype=int
    )
    x = _covariates(rows)
    residuals = np.full((len(rows), len(eligible)), np.nan, dtype=float)
    models: dict[str, dict[str, Any]] = {}
    for column, feature in enumerate(eligible):
        y = np.asarray([row["analysisFeatures"].get(feature, np.nan) for row in rows])
        feature_models: dict[str, Any] = {}
        for fold in range(5):
            train = (folds != fold) & np.isfinite(y)
            test = (folds == fold) & np.isfinite(y)
            if train.sum() == 0:
                raise RuntimeError("empty cross-fitting training fold")
            model = Ridge(alpha=10.0, fit_intercept=True)
            model.fit(x[train], y[train])
            residuals[test, column] = y[test] - model.predict(x[test])
            feature_models[str(fold)] = {
                "coef": [float(value) for value in model.coef_],
                "intercept": float(model.intercept_),
            }
        models[feature] = feature_models
    medians = np.nanmedian(residuals, axis=0)
    mads = np.nanmedian(np.abs(residuals - medians), axis=0)
    scales = np.where(mads > 0, mads, 1.0)
    standardized = (residuals - medians) / scales
    return {
        "schemaVersion": "e07.s10.preprocessing-lock.v1",
        "eligibleFeatures": eligible,
        "featureSupport": support,
        "ridgeAlpha": 10.0,
        "foldRule": "sha256_scenario_family_mod_5",
        "modelsByFeatureAndFold": models,
        "medianByFeature": {
            feature: float(medians[index]) for index, feature in enumerate(eligible)
        },
        "madScaleByFeature": {
            feature: float(scales[index]) for index, feature in enumerate(eligible)
        },
        "unitFallbackFeatures": [
            feature for index, feature in enumerate(eligible) if mads[index] == 0
        ],
        "discoveryStandardized": standardized,
    }


def apply_preprocessing(
    rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> np.ndarray:
    eligible = list(lock["eligibleFeatures"])
    x = _covariates(rows)
    output = np.full((len(rows), len(eligible)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        fold = str(fold_for_family(str(row["scenarioFamilyId"])))
        for column, feature in enumerate(eligible):
            value = row["analysisFeatures"].get(feature)
            if value is None:
                continue
            model = lock["modelsByFeatureAndFold"][feature][fold]
            prediction = float(model["intercept"]) + float(
                np.dot(np.asarray(model["coef"], dtype=float), x[row_index])
            )
            residual = float(value) - prediction
            output[row_index, column] = (
                residual - float(lock["medianByFeature"][feature])
            ) / float(lock["madScaleByFeature"][feature])
    return output


def configuration_profiles(
    rows: Sequence[Mapping[str, Any]], matrix: np.ndarray
) -> tuple[list[str], np.ndarray]:
    candidate_ids = sorted({str(row["candidateId"]) for row in rows})
    profiles = []
    for candidate_id in candidate_ids:
        mask = np.asarray(
            [str(row["candidateId"]) == candidate_id for row in rows], dtype=bool
        )
        profile = np.nanmedian(matrix[mask], axis=0)
        if not np.all(np.isfinite(profile)):
            raise RuntimeError("configuration profile contains missing coordinates")
        profiles.append(profile)
    return candidate_ids, np.asarray(profiles, dtype=float)


def cluster_jaccard(reference: Sequence[int], other: Sequence[int]) -> float:
    reference_labels = sorted(set(map(int, reference)))
    other_labels = sorted(set(map(int, other)))
    scores = np.zeros((len(reference_labels), len(other_labels)), dtype=float)
    for i, left in enumerate(reference_labels):
        a = {index for index, value in enumerate(reference) if int(value) == left}
        for j, right in enumerate(other_labels):
            b = {index for index, value in enumerate(other) if int(value) == right}
            scores[i, j] = len(a & b) / max(1, len(a | b))
    row_index, column_index = linear_sum_assignment(-scores)
    matched = [scores[i, j] for i, j in zip(row_index, column_index)]
    if len(matched) < len(reference_labels):
        matched.extend([0.0] * (len(reference_labels) - len(matched)))
    return float(min(matched)) if matched else 0.0


def _ward(profiles: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(profiles)


def _bootstrap_profiles(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    candidate_ids: Sequence[str],
    sampled_families: Sequence[str],
) -> np.ndarray:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row["candidateId"]), str(row["scenarioFamilyId"]))].append(index)
    result = []
    for candidate_id in candidate_ids:
        indices = [
            index
            for family in sampled_families
            for index in grouped.get((candidate_id, family), [])
        ]
        if not indices:
            raise RuntimeError("bootstrap profile lost a configuration")
        result.append(np.nanmedian(matrix[indices], axis=0))
    return np.asarray(result, dtype=float)


def clustering_discovery(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    *,
    task_id: str,
    status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_ids, profiles = configuration_profiles(rows, matrix)
    families = sorted({str(row["scenarioFamilyId"]) for row in rows})
    observed: dict[int, dict[str, Any]] = {}
    for k in K_GRID:
        labels = _ward(profiles, k)
        aris, jaccards = [], []
        for replicate in range(BOOTSTRAP_REPLICATES):
            rng = np.random.default_rng(
                stable_seed("S10", "cluster-bootstrap", task_id, status, k, replicate)
            )
            sample = [
                families[index]
                for index in rng.integers(0, len(families), len(families))
            ]
            boot_profiles = _bootstrap_profiles(rows, matrix, candidate_ids, sample)
            boot_labels = _ward(boot_profiles, k)
            aris.append(float(adjusted_rand_score(labels, boot_labels)))
            jaccards.append(cluster_jaccard(labels, boot_labels))
        observed[k] = {
            "labels": [int(value) for value in labels],
            "meanBootstrapARI": float(np.mean(aris)),
            "seBootstrapARI": float(np.std(aris, ddof=1) / math.sqrt(len(aris))),
            "meanMinimumClusterJaccard": float(np.mean(jaccards)),
        }
    best_k = max(K_GRID, key=lambda k: (observed[k]["meanBootstrapARI"], -k))
    threshold = (
        observed[best_k]["meanBootstrapARI"] - observed[best_k]["seBootstrapARI"]
    )
    selected_k = min(k for k in K_GRID if observed[k]["meanBootstrapARI"] >= threshold)
    ward_labels = np.asarray(observed[selected_k]["labels"], dtype=int)
    gmm_rows = []
    for k in range(1, 7):
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=20,
            random_state=stable_seed("S10", "gmm", task_id, status, k) % (2**32 - 1),
        ).fit(profiles)
        labels = model.predict(profiles)
        gmm_rows.append(
            {
                "k": k,
                "bic": float(model.bic(profiles)),
                "labels": [int(value) for value in labels],
                "minimumClusterSize": int(min(Counter(labels).values())),
            }
        )
    selected_gmm = min(gmm_rows, key=lambda item: (item["bic"], item["k"]))
    gmm_labels = np.asarray(selected_gmm["labels"], dtype=int)
    cross_ari = float(adjusted_rand_score(ward_labels, gmm_labels))

    # Frozen label-permutation null. Each replicate uses two independently
    # addressed scenario-family bootstraps and takes the maximum over k.
    row_candidate = np.asarray([str(row["candidateId"]) for row in rows], dtype=object)
    family_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family_to_indices[str(row["scenarioFamilyId"])].append(index)
    null_ari, null_jaccard = [], []
    for replicate in range(NULL_REPLICATES):
        rng = np.random.default_rng(
            stable_seed("S10", "cluster-null", task_id, status, replicate)
        )
        permuted = row_candidate.copy()
        for indices in family_to_indices.values():
            values = permuted[indices].copy()
            rng.shuffle(values)
            permuted[indices] = values
        null_rows = [
            {**row, "candidateId": str(permuted[index])}
            for index, row in enumerate(rows)
        ]
        maxima_ari, maxima_j = [], []
        for k in K_GRID:
            sample_a = [
                families[index]
                for index in rng.integers(0, len(families), len(families))
            ]
            sample_b = [
                families[index]
                for index in rng.integers(0, len(families), len(families))
            ]
            left = _ward(
                _bootstrap_profiles(null_rows, matrix, candidate_ids, sample_a), k
            )
            right = _ward(
                _bootstrap_profiles(null_rows, matrix, candidate_ids, sample_b), k
            )
            maxima_ari.append(float(adjusted_rand_score(left, right)))
            maxima_j.append(cluster_jaccard(left, right))
        null_ari.append(max(maxima_ari))
        null_jaccard.append(max(maxima_j))
    ari_q95 = float(np.quantile(null_ari, 0.95, method="higher"))
    jaccard_q95 = float(np.quantile(null_jaccard, 0.95, method="higher"))
    observed_ari = float(observed[selected_k]["meanBootstrapARI"])
    observed_jaccard = float(observed[selected_k]["meanMinimumClusterJaccard"])
    p_ari = (1 + sum(value >= observed_ari for value in null_ari)) / (
        NULL_REPLICATES + 1
    )
    p_j = (1 + sum(value >= observed_jaccard for value in null_jaccard)) / (
        NULL_REPLICATES + 1
    )
    global_p = max(p_ari, p_j)
    minimum_cluster = min(Counter(ward_labels).values())
    gate = (
        observed_ari >= 0.70
        and observed_jaccard >= 0.70
        and observed_ari > ari_q95
        and observed_jaccard > jaccard_q95
        and minimum_cluster >= 2
        and cross_ari >= 0.60
        and selected_gmm["minimumClusterSize"] >= 2
    )
    summary = {
        "schemaVersion": "e07.s10.clustering-result.v1",
        "taskId": task_id,
        "statusStratum": status,
        "candidateIds": candidate_ids,
        "profileSha256": canonical_sha256(
            "E07/S10/configuration-profiles/v1", profiles.tolist()
        ),
        "kResults": {str(key): value for key, value in observed.items()},
        "selectedWardK": selected_k,
        "selectedWardLabels": [int(value) for value in ward_labels],
        "selectedGmm": selected_gmm,
        "crossMethodAdjustedRand": cross_ari,
        "minimumClusterSize": int(minimum_cluster),
        "nullMaxAri95": ari_q95,
        "nullMaxJaccard95": jaccard_q95,
        "rawPValue": float(global_p),
        "preMultiplicityGatePass": bool(gate),
    }
    candidates = []
    if gate:
        for label in sorted(set(ward_labels)):
            members = [
                candidate_ids[index]
                for index, value in enumerate(ward_labels)
                if int(value) == int(label)
            ]
            candidates.append(
                {
                    "methodFamily": "clustering",
                    "taskId": task_id,
                    "statusStratum": status,
                    "machineCandidateKey": canonical_sha256(
                        "E07/S10/cluster-candidate/v1",
                        {
                            "taskId": task_id,
                            "status": status,
                            "members": members,
                        },
                    ),
                    "memberCandidateIds": members,
                    "configurationId": None,
                    "lockedCenterTransition": None,
                    "rawPValue": float(global_p),
                    "effectDirection": "locked_cluster_membership",
                }
            )
    return summary, candidates


def anomaly_discovery(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    *,
    task_id: str,
    status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_ids, profiles = configuration_profiles(rows, matrix)
    model = IsolationForest(
        n_estimators=500,
        max_samples=min(256, len(profiles)),
        max_features=1.0,
        contamination="auto",
        random_state=ANOMALY_SEED % (2**32 - 1),
        n_jobs=1,
    ).fit(profiles)
    scores = -model.decision_function(profiles)
    index = max(range(len(candidate_ids)), key=lambda i: (scores[i], candidate_ids[i]))
    selected = candidate_ids[index]
    observed = float(scores[index])
    family_to_indices: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        family_to_indices[str(row["scenarioFamilyId"])].append(row_index)
    original_ids = np.asarray([str(row["candidateId"]) for row in rows], dtype=object)
    null_max = []
    for replicate in range(NULL_REPLICATES):
        rng = np.random.default_rng(
            stable_seed("S10", "anomaly-null", task_id, status, replicate)
        )
        permuted = original_ids.copy()
        for indices in family_to_indices.values():
            values = permuted[indices].copy()
            rng.shuffle(values)
            permuted[indices] = values
        null_rows = [
            {**row, "candidateId": str(permuted[row_index])}
            for row_index, row in enumerate(rows)
        ]
        _, null_profiles = configuration_profiles(null_rows, matrix)
        null_model = IsolationForest(
            n_estimators=500,
            max_samples=min(256, len(null_profiles)),
            max_features=1.0,
            contamination="auto",
            random_state=ANOMALY_SEED % (2**32 - 1),
            n_jobs=1,
        ).fit(null_profiles)
        null_max.append(float(np.max(-null_model.decision_function(null_profiles))))
    p_value = (1 + sum(value >= observed for value in null_max)) / (NULL_REPLICATES + 1)
    result = {
        "schemaVersion": "e07.s10.anomaly-result.v1",
        "taskId": task_id,
        "statusStratum": status,
        "candidateIds": candidate_ids,
        "profileSha256": canonical_sha256(
            "E07/S10/configuration-profiles/v1", profiles.tolist()
        ),
        "scores": {
            candidate_id: float(score)
            for candidate_id, score in zip(candidate_ids, scores)
        },
        "selectedConfigurationId": selected,
        "selectedScore": observed,
        "medianScore": float(np.median(scores)),
        "nullMaximum95": float(np.quantile(null_max, 0.95, method="higher")),
        "rawPValue": float(p_value),
        "preMultiplicityGatePass": True,
    }
    candidate = {
        "methodFamily": "anomaly",
        "taskId": task_id,
        "statusStratum": status,
        "machineCandidateKey": canonical_sha256(
            "E07/S10/anomaly-candidate/v1",
            {"taskId": task_id, "status": status, "candidateId": selected},
        ),
        "memberCandidateIds": [selected],
        "configurationId": selected,
        "lockedCenterTransition": None,
        "rawPValue": float(p_value),
        "effectDirection": "higher_locked_isolation_score",
    }
    result["model"] = model
    return result, candidate


def row_change_points(row: Mapping[str, Any], shift: int = 0) -> tuple[int, ...]:
    series = row["orderedEventSeries"]
    values = list(
        zip(
            series["proposalCount"],
            series["acceptedCount"],
            series["conflictLosses"],
            map(int, series["stateHashChanged"]),
        )
    )
    if shift:
        values = values[shift:] + values[:shift]
    return exact_change_points(
        values, minimum_segment_length=4, maximum_change_points=3
    )


def _maximum_window_prevalence(
    point_sets: Sequence[Sequence[int]],
) -> tuple[int, float]:
    candidates = range(4, 29)
    prevalence = [
        sum(any(abs(point - center) <= 1 for point in points) for points in point_sets)
        / len(point_sets)
        for center in candidates
    ]
    index = max(range(len(prevalence)), key=lambda i: (prevalence[i], -candidates[i]))
    return candidates[index], float(prevalence[index])


def change_point_discovery(
    rows: Sequence[Mapping[str, Any]], *, task_id: str, status: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, candidates = [], []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidateId"])].append(row)
    for candidate_id in sorted(grouped):
        candidate_rows = sorted(
            grouped[candidate_id], key=lambda row: str(row["scenarioFamilyId"])
        )
        observed_points = [row_change_points(row) for row in candidate_rows]
        center, prevalence = _maximum_window_prevalence(observed_points)
        null_max: list[float] = []
        if prevalence >= 0.60:
            # Precompute every circular shift once, then address the 500 null
            # replicates without changing marginal counts.
            shifted = [
                [row_change_points(row, shift) for shift in range(32)]
                for row in candidate_rows
            ]
            for replicate in range(NULL_REPLICATES):
                rng = np.random.default_rng(
                    stable_seed(
                        "S10",
                        "change-point-null",
                        task_id,
                        status,
                        candidate_id,
                        replicate,
                    )
                )
                points = [
                    shifted[index][int(rng.integers(0, 32))]
                    for index in range(len(candidate_rows))
                ]
                _, null_prevalence = _maximum_window_prevalence(points)
                null_max.append(null_prevalence)
            p_value = (1 + sum(value >= prevalence for value in null_max)) / (
                NULL_REPLICATES + 1
            )
            null_q95 = float(np.quantile(null_max, 0.95, method="higher"))
        else:
            p_value = 1.0
            null_q95 = None
        result = {
            "schemaVersion": "e07.s10.change-point-result.v1",
            "taskId": task_id,
            "statusStratum": status,
            "configurationId": candidate_id,
            "lockedCenterTransition": int(center),
            "discoveryPrevalence": float(prevalence),
            "nullMaximum95": null_q95,
            "rawPValue": float(p_value),
            "preMultiplicityGatePass": bool(
                prevalence >= 0.60 and null_q95 is not None and prevalence > null_q95
            ),
            "changePointCountDistribution": dict(
                sorted(Counter(len(points) for points in observed_points).items())
            ),
        }
        results.append(result)
        if result["preMultiplicityGatePass"]:
            candidates.append(
                {
                    "methodFamily": "change_point",
                    "taskId": task_id,
                    "statusStratum": status,
                    "machineCandidateKey": canonical_sha256(
                        "E07/S10/change-point-candidate/v1",
                        {
                            "taskId": task_id,
                            "status": status,
                            "candidateId": candidate_id,
                            "center": center,
                        },
                    ),
                    "memberCandidateIds": [candidate_id],
                    "configurationId": candidate_id,
                    "lockedCenterTransition": int(center),
                    "rawPValue": float(p_value),
                    "effectDirection": "locked_change_point_window_prevalence",
                }
            )
    return results, candidates


def holm_adjust(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (float(item[1]["rawPValue"]), item[1]["machineCandidateKey"]),
    )
    adjusted = [1.0] * len(candidates)
    running = 0.0
    total = len(candidates)
    for rank, (original_index, candidate) in enumerate(ordered):
        value = min(1.0, (total - rank) * float(candidate["rawPValue"]))
        running = max(running, value)
        adjusted[original_index] = running
    return [
        {
            **dict(candidate),
            "holmAdjustedPValue": float(adjusted[index]),
            "discoveryMultiplicityPass": bool(adjusted[index] <= 0.05),
        }
        for index, candidate in enumerate(candidates)
    ]


def confound_audit(
    rows: Sequence[Mapping[str, Any]], matrix: np.ndarray
) -> dict[str, Any]:
    lengths = np.asarray(
        [float(row["confounds"]["observedEventLength"]) for row in rows]
    )
    correlations = []
    for column in range(matrix.shape[1]):
        if np.std(lengths) == 0 or np.nanstd(matrix[:, column]) == 0:
            correlations.append(0.0)
        else:
            value = spearmanr(lengths, matrix[:, column], nan_policy="omit").statistic
            correlations.append(0.0 if not np.isfinite(value) else float(value))
    statuses = np.asarray([str(row["statusStratum"]) for row in rows])
    if len(set(statuses)) < 2:
        status_ba = None
        status_pass = True
    else:
        predicted, actual = [], []
        folds = np.asarray(
            [fold_for_family(str(row["scenarioFamilyId"])) for row in rows]
        )
        for fold in range(5):
            train, test = folds != fold, folds == fold
            model = LogisticRegression(max_iter=1000, random_state=0)
            model.fit(matrix[train], statuses[train])
            predicted.extend(model.predict(matrix[test]))
            actual.extend(statuses[test])
        status_ba = float(balanced_accuracy_score(actual, predicted))
        status_pass = status_ba <= 0.65
    maximum = max(map(abs, correlations), default=0.0)
    return {
        "maximumAbsoluteSpearmanWithLength": float(maximum),
        "lengthThreshold": 0.20,
        "lengthPass": maximum <= 0.20,
        "statusBalancedAccuracy": status_ba,
        "statusThreshold": 0.65,
        "statusPass": status_pass,
        "taskPrediction": "not_applicable_no_cross_task_pooling",
        "pass": maximum <= 0.20 and status_pass,
    }


def discovery_analysis(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_results: dict[str, Any] = {
        "preprocessing": {},
        "confounds": {},
        "clustering": [],
        "anomaly": [],
        "changePoint": [],
    }
    machine_candidates: list[dict[str, Any]] = []
    locks: dict[str, Any] = {}
    for task_id in SPATIAL_TASKS:
        task_rows = [row for row in rows if row["taskId"] == task_id]
        statuses = sorted({str(row["statusStratum"]) for row in task_rows})
        for status in statuses:
            stratum = [row for row in task_rows if row["statusStratum"] == status]
            if len(stratum) < 8:
                all_results.setdefault("rareStatusStrata", []).append(
                    {
                        "taskId": task_id,
                        "statusStratum": status,
                        "rowCount": len(stratum),
                        "analysis": "descriptive_only",
                    }
                )
                continue
            lock = fit_preprocessing(stratum)
            matrix = np.asarray(lock.pop("discoveryStandardized"), dtype=float)
            key = f"{task_id}::{status}"
            audit = confound_audit(stratum, matrix)
            if not audit["pass"]:
                raise RuntimeError(f"frozen confound gate failed for {key}: {audit}")
            cluster_result, cluster_candidates = clustering_discovery(
                stratum, matrix, task_id=task_id, status=status
            )
            anomaly_result, anomaly_candidate = anomaly_discovery(
                stratum, matrix, task_id=task_id, status=status
            )
            anomaly_model = anomaly_result.pop("model")
            cp_results, cp_candidates = change_point_discovery(
                stratum, task_id=task_id, status=status
            )
            all_results["preprocessing"][key] = lock
            all_results["confounds"][key] = audit
            all_results["clustering"].append(cluster_result)
            all_results["anomaly"].append(anomaly_result)
            all_results["changePoint"].extend(cp_results)
            machine_candidates.extend(cluster_candidates)
            machine_candidates.append(anomaly_candidate)
            machine_candidates.extend(cp_candidates)
            candidate_ids, profiles = configuration_profiles(stratum, matrix)
            locks[key] = {
                "preprocessing": lock,
                "candidateIds": candidate_ids,
                "profiles": profiles.tolist(),
                "cluster": cluster_result,
                "anomalyModel": anomaly_model,
                "anomaly": anomaly_result,
            }
    adjusted = holm_adjust(machine_candidates)
    passing = [row for row in adjusted if row["discoveryMultiplicityPass"]]
    catalog = {
        "schemaVersion": "e07.s10.discovery-analysis.v1",
        "machineMultiplicityFamilySize": len(adjusted),
        "machineCandidates": adjusted,
        "discoveryPassingCandidates": passing,
        "results": all_results,
    }
    lock_document = {
        "schemaVersion": "e07.s10.discovery-lock.v1",
        "researchStepId": "S10",
        "candidateDefinitions": adjusted,
        "discoveryPassingCandidateKeys": sorted(
            row["machineCandidateKey"] for row in passing
        ),
        "featureSupportMasksAndPreprocessing": {
            key: value["preprocessing"] for key, value in locks.items()
        },
        "configurationProfiles": {
            key: {
                "candidateIds": value["candidateIds"],
                "profiles": value["profiles"],
            }
            for key, value in locks.items()
        },
        "clusterRepresentatives": {
            key: value["cluster"] for key, value in locks.items()
        },
        "anomalyThresholds": {key: value["anomaly"] for key, value in locks.items()},
        "changePointWindows": [
            {
                "machineCandidateKey": row["machineCandidateKey"],
                "center": row["lockedCenterTransition"],
            }
            for row in adjusted
            if row["methodFamily"] == "change_point"
        ],
        "multiplicityFamily": {
            "method": "Holm",
            "alpha": 0.05,
            "size": len(adjusted),
            "candidateKeys": sorted(row["machineCandidateKey"] for row in adjusted),
        },
        "reproductionRefitPermitted": False,
    }
    lock_document["discoveryLockSha256"] = canonical_sha256(
        "E07/S10/discovery-lock/v1", lock_document
    )
    # Runtime-only frozen estimators remain in memory and are never used to
    # allocate scenarios or mutate an archive.
    lock_document["_runtimeLocks"] = locks
    return catalog, lock_document


def _locked_cluster_assignment(
    discovery_profiles: np.ndarray,
    labels: Sequence[int],
    reproduction_profiles: np.ndarray,
) -> np.ndarray:
    centroids = {
        label: discovery_profiles[np.asarray(labels) == label].mean(axis=0)
        for label in sorted(set(labels))
    }
    return np.asarray(
        [
            min(
                centroids,
                key=lambda label: (
                    float(np.linalg.norm(row - centroids[label])),
                    int(label),
                ),
            )
            for row in reproduction_profiles
        ],
        dtype=int,
    )


def reproduction_analysis(
    rows: Sequence[Mapping[str, Any]],
    discovery_catalog: Mapping[str, Any],
    discovery_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    passing = {
        row["machineCandidateKey"]: row
        for row in discovery_catalog["discoveryPassingCandidates"]
    }
    runtime_locks = discovery_lock["_runtimeLocks"]
    results = []
    for key, candidate in sorted(passing.items()):
        lock_key = f"{candidate['taskId']}::{candidate['statusStratum']}"
        task_rows = [
            row
            for row in rows
            if row["taskId"] == candidate["taskId"]
            and row["statusStratum"] == candidate["statusStratum"]
        ]
        if not task_rows:
            results.append(
                {
                    **candidate,
                    "reproductionPass": False,
                    "failureReason": "locked_status_stratum_absent",
                }
            )
            continue
        lock = runtime_locks[lock_key]
        matrix = apply_preprocessing(task_rows, lock["preprocessing"])
        candidate_ids, profiles = configuration_profiles(task_rows, matrix)
        if candidate_ids != lock["candidateIds"]:
            results.append(
                {
                    **candidate,
                    "reproductionPass": False,
                    "failureReason": "configuration_population_changed",
                }
            )
            continue
        if candidate["methodFamily"] == "clustering":
            labels = lock["cluster"]["selectedWardLabels"]
            assigned = _locked_cluster_assignment(
                np.asarray(lock["profiles"], dtype=float), labels, profiles
            )
            member_set = set(candidate["memberCandidateIds"])
            discovery_label = next(
                labels[index]
                for index, item in enumerate(candidate_ids)
                if item in member_set
            )
            reproduced_set = {
                candidate_ids[index]
                for index, label in enumerate(assigned)
                if int(label) == int(discovery_label)
            }
            jaccard = len(member_set & reproduced_set) / max(
                1, len(member_set | reproduced_set)
            )
            passed = jaccard >= 0.70
            results.append(
                {
                    **candidate,
                    "reproductionMetric": "locked_membership_Jaccard",
                    "reproductionValue": float(jaccard),
                    "reproductionThreshold": 0.70,
                    "reproductionPass": bool(passed),
                    "reproducedMemberCandidateIds": sorted(reproduced_set),
                }
            )
        elif candidate["methodFamily"] == "anomaly":
            model: IsolationForest = lock["anomalyModel"]
            scores = -model.decision_function(profiles)
            index = candidate_ids.index(str(candidate["configurationId"]))
            score = float(scores[index])
            same_direction = score > float(np.median(scores))
            original_ids = np.asarray(
                [str(row["candidateId"]) for row in task_rows], dtype=object
            )
            family_indices: dict[str, list[int]] = defaultdict(list)
            for row_index, row in enumerate(task_rows):
                family_indices[str(row["scenarioFamilyId"])].append(row_index)
            null_max = []
            for replicate in range(NULL_REPLICATES):
                rng = np.random.default_rng(
                    stable_seed("S10", "reproduction-anomaly-null", key, replicate)
                )
                permuted = original_ids.copy()
                for indices in family_indices.values():
                    values = permuted[indices].copy()
                    rng.shuffle(values)
                    permuted[indices] = values
                null_rows = [
                    {**row, "candidateId": str(permuted[row_index])}
                    for row_index, row in enumerate(task_rows)
                ]
                _, null_profiles = configuration_profiles(null_rows, matrix)
                null_max.append(float(np.max(-model.decision_function(null_profiles))))
            p_value = (1 + sum(value >= score for value in null_max)) / (
                NULL_REPLICATES + 1
            )
            results.append(
                {
                    **candidate,
                    "reproductionMetric": "locked_isolation_score",
                    "reproductionValue": score,
                    "sameDirection": bool(same_direction),
                    "reproductionRawPValue": float(p_value),
                    "reproductionPassPreHolm": bool(same_direction),
                }
            )
        else:
            candidate_rows = [
                row
                for row in task_rows
                if row["candidateId"] == candidate["configurationId"]
            ]
            points = [row_change_points(row) for row in candidate_rows]
            center = int(candidate["lockedCenterTransition"])
            prevalence = sum(
                any(abs(point - center) <= 1 for point in row_points)
                for row_points in points
            ) / len(points)
            results.append(
                {
                    **candidate,
                    "reproductionMetric": "locked_window_prevalence",
                    "reproductionValue": float(prevalence),
                    "reproductionThreshold": 0.50,
                    "reproductionPass": bool(prevalence >= 0.50),
                }
            )
    anomaly_indices = [
        index for index, row in enumerate(results) if row["methodFamily"] == "anomaly"
    ]
    anomaly_candidates = [
        {
            "machineCandidateKey": results[index]["machineCandidateKey"],
            "rawPValue": results[index]["reproductionRawPValue"],
        }
        for index in anomaly_indices
    ]
    anomaly_adjusted = {
        row["machineCandidateKey"]: row for row in holm_adjust(anomaly_candidates)
    }
    for index in anomaly_indices:
        row = results[index]
        adjusted = anomaly_adjusted[row["machineCandidateKey"]]
        row["reproductionHolmAdjustedPValue"] = adjusted["holmAdjustedPValue"]
        row["reproductionPass"] = bool(
            row["reproductionPassPreHolm"] and adjusted["holmAdjustedPValue"] <= 0.05
        )
    return results


def serializable_discovery_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lock.items() if key != "_runtimeLocks"}


def selected_exemplars(
    discovery_rows: Sequence[Mapping[str, Any]],
    reproduced: Sequence[Mapping[str, Any]],
    discovery_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Mechanically select six blinded exemplars per reproduced candidate."""

    packets = []
    runtime_locks = discovery_lock["_runtimeLocks"]
    for candidate in reproduced:
        if not candidate.get("reproductionPass"):
            continue
        lock_key = f"{candidate['taskId']}::{candidate['statusStratum']}"
        lock = runtime_locks[lock_key]
        rows = [
            row
            for row in discovery_rows
            if row["taskId"] == candidate["taskId"]
            and row["statusStratum"] == candidate["statusStratum"]
        ]
        matrix = apply_preprocessing(rows, lock["preprocessing"])
        members = set(candidate["memberCandidateIds"])
        member_indices = [
            index for index, row in enumerate(rows) if row["candidateId"] in members
        ]
        null_indices = [
            index for index, row in enumerate(rows) if row["candidateId"] not in members
        ]
        center = np.nanmedian(matrix[member_indices], axis=0)
        distances = np.linalg.norm(matrix - center, axis=1)
        nearest = sorted(
            member_indices, key=lambda i: (distances[i], rows[i]["logicalResultSha256"])
        )[:2]
        extremes = sorted(
            member_indices,
            key=lambda i: (-distances[i], rows[i]["logicalResultSha256"]),
        )[:2]
        controls = sorted(
            null_indices, key=lambda i: (distances[i], rows[i]["logicalResultSha256"])
        )[:2]
        chosen = [
            *[("nearest_locked_medoid", index) for index in nearest],
            *[("locked_extreme", index) for index in extremes],
            *[("matched_null_control", index) for index in controls],
        ]
        if len(chosen) != 6:
            raise RuntimeError("could not construct six frozen blinded exemplars")
        for ordinal, (role, index) in enumerate(chosen):
            row = rows[index]
            body = {
                "schemaVersion": "e07.s10.blinded-exemplar.v1",
                "machineCandidateKeyBlinded": canonical_sha256(
                    "E07/S10/blinded-candidate/v1",
                    candidate["machineCandidateKey"],
                )[:16],
                "blindedExemplarId": canonical_sha256(
                    "E07/S10/blinded-exemplar/v1",
                    [
                        candidate["machineCandidateKey"],
                        row["logicalResultSha256"],
                        ordinal,
                    ],
                ),
                "exemplarRole": role,
                "taskContract": row["taskId"],
                "nativeStatus": {
                    "stopReason": row["stopReason"],
                    "failed": row["failed"],
                    "censored": row["censored"],
                    "horizon": row["confounds"]["nativeHorizon"],
                },
                "eventSummaryFeatures": row["analysisFeatures"],
                "orderedEventSeries": row["orderedEventSeries"],
                "traceAvailability": row["confounds"]["traceAvailability"],
                "traceSelectionReason": row["confounds"]["traceSelectionReason"],
                "configurationId": None,
                "candidateRole": None,
                "objectiveValues": None,
                "validationValues": None,
                "policySource": None,
                "s07Arm": None,
                "reviewerAnnotation": None,
            }
            packets.append(body)
    return packets


def flatten_feature_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    result = []
    for row in rows:
        base = {
            "logicalOrdinal": row["logicalOrdinal"],
            "logicalReservationId": row["logicalReservationId"],
            "phase": row["phase"],
            "taskId": row["taskId"],
            "scenarioFamilyOrdinal": row["scenarioFamilyOrdinal"],
            "scenarioFamilyId": row["scenarioFamilyId"],
            "candidateId": row["candidateId"],
            "candidateRole": row["candidateRole"],
            "stopReason": row["stopReason"],
            "failed": row["failed"],
            "censored": row["censored"],
            "statusStratum": row["statusStratum"],
            "logicalResultSha256": row["logicalResultSha256"],
            "featureRecordSha256": row["featureRecordSha256"],
            "orderedSummaryCommitmentSha256": row["orderedSummaryCommitmentSha256"],
        }
        base.update(row["analysisFeatures"])
        result.append(base)
    return pd.DataFrame(result)


def flatten_series_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "logicalOrdinal": row["logicalOrdinal"],
                "logicalReservationId": row["logicalReservationId"],
                "phase": row["phase"],
                "taskId": row["taskId"],
                "scenarioFamilyId": row["scenarioFamilyId"],
                "candidateId": row["candidateId"],
                "statusStratum": row["statusStratum"],
                "proposalCount": row["orderedEventSeries"]["proposalCount"],
                "acceptedCount": row["orderedEventSeries"]["acceptedCount"],
                "conflictLosses": row["orderedEventSeries"]["conflictLosses"],
                "invalidProposals": row["orderedEventSeries"]["invalidProposals"],
                "stateHashChanged": row["orderedEventSeries"]["stateHashChanged"],
                "orderedSummaryCommitmentSha256": row["orderedSummaryCommitmentSha256"],
            }
            for row in rows
        ]
    )


def result_integrity(
    rows: Sequence[Mapping[str, Any]], expected: int
) -> dict[str, Any]:
    logical_ids = [str(row["logicalReservationId"]) for row in rows]
    result_hashes = [str(row["logicalResultSha256"]) for row in rows]
    return {
        "expectedLogicalRows": expected,
        "observedLogicalRows": len(rows),
        "uniqueLogicalReservationIds": len(set(logical_ids)),
        "uniqueLogicalResultSha256": len(set(result_hashes)),
        "replayPassRows": sum(bool(row["replayPass"]) for row in rows),
        "nativeContractPassRows": sum(bool(row["nativeContractPass"]) for row in rows),
        "completeTrajectoryReconstructions": sum(
            bool(row["completeTrajectoryReconstructed"]) for row in rows
        ),
        "featureOutcomePlaneReads": sum(
            bool(row["outcomePlaneReadForFeatures"]) for row in rows
        ),
        "digestNatural": rows_digest(rows),
        "digestReverse": rows_digest(list(reversed(rows))),
        "pass": (
            len(rows) == expected
            and len(set(logical_ids)) == expected
            and len(set(result_hashes)) == expected
            and all(row["replayPass"] and row["nativeContractPass"] for row in rows)
            and not any(row["completeTrajectoryReconstructed"] for row in rows)
            and not any(row["outcomePlaneReadForFeatures"] for row in rows)
            and rows_digest(rows) == rows_digest(list(reversed(rows)))
        ),
    }
