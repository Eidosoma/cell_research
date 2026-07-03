#!/usr/bin/env python3
"""Train and evaluate E07 S05 behavior-predictor baselines and neural surrogate."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.predictor_schema import (  # noqa: E402
    PREDICTOR_SCHEMA_VERSION,
    SPLIT_DEFINITIONS,
    assign_group_holdout,
    calibration_bins,
    metrics_for_predictions,
    split_summary,
    transformed_target,
    validate_prediction_artifacts,
)


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703

CORPUS_COLUMNS = (
    "corpus_row_id",
    "source_experiment_id",
    "source_table",
    "source_metric_name",
    "metric_value",
    "metric_unit",
    "metric_direction",
    "metric_family",
    "world_id",
    "world_link_status",
    "canonical_policy_id",
    "policy_uid",
    "policy_link_status",
    "canonical_goal_id",
    "goal_uid",
    "goal_link_status",
    "perturbation_type",
    "evidence_kind",
)

CATEGORICAL_COLUMNS = (
    "source_experiment_id",
    "source_table",
    "source_metric_name",
    "metric_unit",
    "metric_direction",
    "metric_family",
    "world_id",
    "world_family",
    "substrate_kind",
    "world_record_granularity",
    "world_completeness",
    "canonical_policy_id",
    "policy_source_experiment_id",
    "policy_family",
    "policy_algorithm",
    "policy_representation_type",
    "policy_abstraction_kind",
    "policy_information_access",
    "policy_execution_backend",
    "policy_direction_support",
    "canonical_goal_id",
    "goal_source_experiment_id",
    "goal_family",
    "goal_kind",
    "goal_abstraction_kind",
    "goal_target_direction",
    "goal_unit",
    "goal_metric_family",
    "goal_representation_status",
    "goal_conflict_group_id",
    "perturbation_type",
    "world_link_status",
    "policy_link_status",
    "goal_link_status",
)

NUMERIC_COLUMNS = (
    "has_world_link",
    "has_policy_link",
    "has_goal_link",
    "policy_metadata_only",
    "policy_requires_memory",
    "policy_requires_signaling",
    "policy_uses_global_oracle",
    "goal_partial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s04-corpus", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "results" / "e07_unified_behavior_corpus.parquet")
    parser.add_argument("--s01-world-inventory", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s02-policy-table", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s03-goal-table", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_goal_representations.parquet")
    parser.add_argument("--max-rows-per-source-metric", type=int, default=1500)
    parser.add_argument("--max-linear-train-rows", type=int, default=250000)
    parser.add_argument("--max-neural-train-rows", type=int, default=120000)
    parser.add_argument("--neural-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = __import__("hashlib").sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def self_referential_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "artifactType": "file",
        "note": "Checksum omitted because this report or manifest contains the artifact list.",
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def boolish(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    text = clean_text(value).lower()
    return 1.0 if text in {"true", "1", "yes"} else 0.0


def deterministic_cap(frame: pd.DataFrame, max_rows: int, *, salt: str) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    hashes = pd.util.hash_pandas_object(frame["corpus_row_id"].astype(str) + f"|{salt}", index=False)
    return frame.assign(_cap_hash=hashes).sort_values("_cap_hash", kind="mergesort").head(max_rows).drop(columns=["_cap_hash"])


def stratified_modeling_sample(corpus: pd.DataFrame, max_per_stratum: int) -> pd.DataFrame:
    frame = corpus.copy()
    frame["_sample_hash"] = pd.util.hash_pandas_object(frame["corpus_row_id"].astype(str), index=False).astype("uint64")
    frame["_stratum"] = frame["source_experiment_id"].astype(str) + "|" + frame["source_table"].astype(str) + "|" + frame["source_metric_name"].astype(str)
    sampled = (
        frame.sort_values(["_stratum", "_sample_hash"], kind="mergesort")
        .groupby("_stratum", sort=False, group_keys=False)
        .head(max_per_stratum)
        .drop(columns=["_sample_hash", "_stratum"])
        .reset_index(drop=True)
    )
    return sampled


def load_feature_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    corpus = pd.read_parquet(args.s04_corpus, columns=list(CORPUS_COLUMNS))
    corpus = corpus[corpus["evidence_kind"].astype(str) == "metric_observation"].copy()
    full_stats = {
        "s04CorpusRows": int(len(corpus)),
        "s04CorpusSha256": sha256_file(args.s04_corpus),
        "rowsBySourceExperiment": corpus.groupby("source_experiment_id").size().astype(int).to_dict(),
        "sourceMetricCount": int(corpus["source_metric_name"].nunique()),
        "worldGroupCount": int(corpus.loc[corpus["world_id"].astype(str) != "", "world_id"].nunique()),
        "policyGroupCount": int(corpus.loc[corpus["canonical_policy_id"].astype(str) != "", "canonical_policy_id"].nunique()),
        "goalGroupCount": int(corpus.loc[corpus["canonical_goal_id"].astype(str) != "", "canonical_goal_id"].nunique()),
        "perturbationGroupCount": int(corpus.loc[corpus["perturbation_type"].astype(str) != "", "perturbation_type"].nunique()),
    }
    modeling = stratified_modeling_sample(corpus, args.max_rows_per_source_metric)
    full_stats["modelingRows"] = int(len(modeling))
    full_stats["modelingRowsBySourceExperiment"] = modeling.groupby("source_experiment_id").size().astype(int).to_dict()

    world = pd.read_csv(args.s01_world_inventory)[
        ["world_id", "world_family", "substrate_kind", "record_granularity", "completeness"]
    ].rename(columns={"record_granularity": "world_record_granularity", "completeness": "world_completeness"})
    policy = pd.read_parquet(args.s02_policy_table)[
        [
            "canonical_policy_id",
            "source_experiment_id",
            "policy_family",
            "algorithm",
            "representation_type",
            "abstraction_kind",
            "metadata_only",
            "requires_memory",
            "requires_signaling",
            "uses_global_oracle",
            "information_access",
            "execution_backend",
            "direction_support",
        ]
    ].drop_duplicates("canonical_policy_id")
    policy = policy.rename(
        columns={
            "source_experiment_id": "policy_source_experiment_id",
            "algorithm": "policy_algorithm",
            "representation_type": "policy_representation_type",
            "abstraction_kind": "policy_abstraction_kind",
            "information_access": "policy_information_access",
            "execution_backend": "policy_execution_backend",
            "direction_support": "policy_direction_support",
        }
    )
    goal = pd.read_parquet(args.s03_goal_table)[
        [
            "canonical_goal_id",
            "source_experiment_id",
            "goal_family",
            "goal_kind",
            "abstraction_kind",
            "representation_status",
            "partial",
            "target_direction",
            "unit",
            "metric_family",
            "conflict_group_id",
        ]
    ].drop_duplicates("canonical_goal_id")
    goal = goal.rename(
        columns={
            "source_experiment_id": "goal_source_experiment_id",
            "abstraction_kind": "goal_abstraction_kind",
            "target_direction": "goal_target_direction",
            "unit": "goal_unit",
            "metric_family": "goal_metric_family",
            "representation_status": "goal_representation_status",
            "partial": "goal_partial",
            "conflict_group_id": "goal_conflict_group_id",
        }
    )
    frame = modeling.merge(world, on="world_id", how="left").merge(policy, on="canonical_policy_id", how="left").merge(goal, on="canonical_goal_id", how="left")
    frame["target_transformed"] = [
        transformed_target(value, direction) for value, direction in zip(frame["metric_value"], frame["metric_direction"], strict=False)
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_transformed"]).reset_index(drop=True)
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).map(lambda text: text if text else "__missing__")
    frame["has_world_link"] = (frame["world_id"] != "__missing__").astype(float)
    frame["has_policy_link"] = (frame["canonical_policy_id"] != "__missing__").astype(float)
    frame["has_goal_link"] = (frame["canonical_goal_id"] != "__missing__").astype(float)
    for column in ("metadata_only", "requires_memory", "requires_signaling", "uses_global_oracle", "goal_partial"):
        if column not in frame.columns:
            frame[column] = 0.0
    frame["policy_metadata_only"] = frame["metadata_only"].map(boolish).astype(float)
    frame["policy_requires_memory"] = frame["requires_memory"].map(boolish).astype(float)
    frame["policy_requires_signaling"] = frame["requires_signaling"].map(boolish).astype(float)
    frame["policy_uses_global_oracle"] = frame["uses_global_oracle"].map(boolish).astype(float)
    frame["goal_partial"] = frame["goal_partial"].map(boolish).astype(float)
    return frame, full_stats


def feature_dicts(frame: pd.DataFrame) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for row in frame[list(CATEGORICAL_COLUMNS) + list(NUMERIC_COLUMNS)].to_dict(orient="records"):
        features: dict[str, float] = {}
        for column in CATEGORICAL_COLUMNS:
            features[f"{column}={row[column]}"] = 1.0
        for column in NUMERIC_COLUMNS:
            features[f"num:{column}"] = float(row[column])
        records.append(features)
    return records


def source_metric_median_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    group_cols = ["source_experiment_id", "source_metric_name", "metric_direction"]
    global_median = float(train["target_transformed"].median())
    medians = train.groupby(group_cols)["target_transformed"].median().to_dict()
    preds = []
    for row in test[group_cols].to_dict(orient="records"):
        key = tuple(row[column] for column in group_cols)
        preds.append(float(medians.get(key, global_median)))
    return np.asarray(preds, dtype=float)


def fit_hashed_linear(train: pd.DataFrame, test: pd.DataFrame, *, max_train_rows: int, split_name: str) -> tuple[np.ndarray, FeatureHasher, SGDRegressor]:
    train_fit = deterministic_cap(train, max_train_rows, salt=f"{split_name}:linear")
    hasher = FeatureHasher(n_features=2**16, input_type="dict", alternate_sign=False)
    x_train = hasher.transform(feature_dicts(train_fit))
    y_train = train_fit["target_transformed"].to_numpy(dtype=float)
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    x_test = hasher.transform(feature_dicts(test))
    return model.predict(x_test), hasher, model


def build_vocabs(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = sorted(set(train[column].astype(str)))
        vocabs[column] = {"__UNK__": 0, **{value: index + 1 for index, value in enumerate(values)}}
    return vocabs


def encode_categories(frame: pd.DataFrame, vocabs: Mapping[str, Mapping[str, int]]) -> np.ndarray:
    columns = []
    for column in CATEGORICAL_COLUMNS:
        vocab = vocabs[column]
        columns.append(frame[column].astype(str).map(lambda value: vocab.get(value, 0)).to_numpy(dtype=np.int64))
    return np.stack(columns, axis=1)


def encode_numeric(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(NUMERIC_COLUMNS)].astype(float).to_numpy(dtype=np.float32)


class EmbeddingMLP(torch.nn.Module):
    def __init__(self, vocab_sizes: Sequence[int], numeric_dim: int) -> None:
        super().__init__()
        dims = [min(32, max(4, int(round(size**0.25 * 4)))) for size in vocab_sizes]
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(size, dim) for size, dim in zip(vocab_sizes, dims, strict=True)])
        input_dim = int(sum(dims) + numeric_dim)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        embedded = [emb(cat[:, index]) for index, emb in enumerate(self.embeddings)]
        joined = torch.cat([*embedded, num], dim=1)
        return self.network(joined).squeeze(1)


def neural_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    split_name: str,
    max_train_rows: int,
    epochs: int,
    batch_size: int,
    model_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_fit = deterministic_cap(train, max_train_rows, salt=f"{split_name}:neural")
    vocabs = build_vocabs(train_fit)
    x_cat = encode_categories(train_fit, vocabs)
    x_num = encode_numeric(train_fit)
    y = train_fit["target_transformed"].to_numpy(dtype=np.float32)
    vocab_sizes = [len(vocabs[column]) for column in CATEGORICAL_COLUMNS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingMLP(vocab_sizes, len(NUMERIC_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    indices = np.arange(len(train_fit))
    losses: list[float] = []
    for epoch in range(epochs):
        np.random.default_rng(RANDOM_SEED + epoch).shuffle(indices)
        batch_losses = []
        model.train()
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            cat_tensor = torch.as_tensor(x_cat[batch_idx], dtype=torch.long, device=device)
            num_tensor = torch.as_tensor(x_num[batch_idx], dtype=torch.float32, device=device)
            y_tensor = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(cat_tensor, num_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))

    model.eval()
    test_cat = encode_categories(test, vocabs)
    test_num = encode_numeric(test)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            cat_tensor = torch.as_tensor(test_cat[start : start + batch_size], dtype=torch.long, device=device)
            num_tensor = torch.as_tensor(test_num[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(cat_tensor, num_tensor).detach().cpu().numpy())
    model_path = model_dir / f"neural_embedding_mlp_{split_name}.pt"
    vocab_path = model_dir / f"neural_vocab_{split_name}.json"
    torch.save(
        {
            "schemaVersion": PREDICTOR_SCHEMA_VERSION,
            "splitName": split_name,
            "stateDict": model.state_dict(),
            "categoricalColumns": list(CATEGORICAL_COLUMNS),
            "numericColumns": list(NUMERIC_COLUMNS),
            "vocabSizes": vocab_sizes,
            "lossHistory": losses,
            "device": str(device),
        },
        model_path,
    )
    write_json(vocab_path, vocabs)
    metadata = {
        "modelPath": str(model_path),
        "vocabPath": str(vocab_path),
        "device": str(device),
        "trainingRows": int(len(train_fit)),
        "epochs": int(epochs),
        "batchSize": int(batch_size),
        "lossHistory": losses,
    }
    return np.concatenate(preds), metadata


def evaluate_predictions(
    *,
    split_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_rows: int,
    test_rows: int,
    model_kind: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = metrics_for_predictions(y_true, y_pred)
    bins = calibration_bins(y_true, y_pred, n_bins=10)
    calibration_ece = float(np.average(bins["abs_gap"], weights=bins["row_count"])) if not bins.empty else float("nan")
    row = {
        "split_name": split_name,
        "model_name": model_name,
        "model_kind": model_kind,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "calibration_ece": calibration_ece,
        **metrics,
    }
    bins.insert(0, "model_name", model_name)
    bins.insert(0, "split_name", split_name)
    return row, bins


def evaluate_split(
    frame: pd.DataFrame,
    *,
    split_name: str,
    group_column: str,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    test_mask = assign_group_holdout(frame, group_column=group_column, split_name=split_name, test_fraction=0.2)
    group_values = frame[group_column].fillna("").astype(str)
    eligible = group_values != "__missing__"
    train = frame[eligible & ~test_mask].reset_index(drop=True)
    test = frame[eligible & test_mask].reset_index(drop=True)
    summary = split_summary(frame.assign(**{group_column: group_values.where(group_values != "__missing__", "")}), split_name=split_name, group_column=group_column, test_mask=test_mask)
    metrics_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    y_true = test["target_transformed"].to_numpy(dtype=float)

    global_pred = np.repeat(float(train["target_transformed"].median()), len(test))
    row, bins = evaluate_predictions(
        split_name=split_name,
        model_name="global_median",
        y_true=y_true,
        y_pred=global_pred,
        train_rows=len(train),
        test_rows=len(test),
        model_kind="baseline",
    )
    metrics_rows.append(row)
    calibration_rows.append(bins)

    metric_pred = source_metric_median_predictions(train, test)
    row, bins = evaluate_predictions(
        split_name=split_name,
        model_name="source_metric_median",
        y_true=y_true,
        y_pred=metric_pred,
        train_rows=len(train),
        test_rows=len(test),
        model_kind="baseline",
    )
    metrics_rows.append(row)
    calibration_rows.append(bins)

    linear_pred, hasher, linear_model = fit_hashed_linear(train, test, max_train_rows=args.max_linear_train_rows, split_name=split_name)
    row, bins = evaluate_predictions(
        split_name=split_name,
        model_name="hashed_linear_sgd",
        y_true=y_true,
        y_pred=linear_pred,
        train_rows=min(len(train), args.max_linear_train_rows),
        test_rows=len(test),
        model_kind="linear_surrogate",
    )
    metrics_rows.append(row)
    calibration_rows.append(bins)
    linear_path = model_dir / f"hashed_linear_sgd_{split_name}.pkl"
    with linear_path.open("wb") as handle:
        pickle.dump({"hasher": hasher, "model": linear_model, "categoricalColumns": CATEGORICAL_COLUMNS, "numericColumns": NUMERIC_COLUMNS}, handle)
    model_records.append({"splitName": split_name, "modelName": "hashed_linear_sgd", "path": str(linear_path)})

    neural_pred, neural_metadata = neural_predict(
        train,
        test,
        split_name=split_name,
        max_train_rows=args.max_neural_train_rows,
        epochs=args.neural_epochs,
        batch_size=args.batch_size,
        model_dir=model_dir,
    )
    row, bins = evaluate_predictions(
        split_name=split_name,
        model_name="neural_embedding_mlp",
        y_true=y_true,
        y_pred=neural_pred,
        train_rows=int(neural_metadata["trainingRows"]),
        test_rows=len(test),
        model_kind="neural_surrogate",
    )
    metrics_rows.append(row)
    calibration_rows.append(bins)
    model_records.append({"splitName": split_name, "modelName": "neural_embedding_mlp", **neural_metadata})

    return metrics_rows, calibration_rows, summary, model_records


def plot_performance(metrics: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    model_order = ["global_median", "source_metric_median", "hashed_linear_sgd", "neural_embedding_mlp"]
    split_order = list(SPLIT_DEFINITIONS.keys())
    pivot = metrics.pivot(index="split_name", columns="model_name", values="mae").reindex(split_order)[model_order]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.18
    for idx, model_name in enumerate(model_order):
        ax.bar(x + (idx - 1.5) * width, pivot[model_name].to_numpy(), width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("MAE on oriented signed-log target")
    ax.set_title("E07 S05 held-out behavior-predictor performance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).copy()
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.to_dict(orient="records"):
        values = [clean_text(record.get(column)).replace("|", "\\|").replace("\n", " ") for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def result_classification(metrics: pd.DataFrame, validation: pd.DataFrame) -> tuple[str, int, pd.DataFrame]:
    rows = []
    for split_name, split_metrics in metrics.groupby("split_name"):
        baseline = split_metrics.loc[split_metrics["model_name"] == "source_metric_median", "mae"]
        learned = split_metrics[split_metrics["model_name"].isin(["hashed_linear_sgd", "neural_embedding_mlp"])]
        if baseline.empty or learned.empty:
            continue
        best = learned.loc[learned["mae"].idxmin()]
        baseline_mae = float(baseline.iloc[0])
        improvement = float((baseline_mae - best["mae"]) / baseline_mae) if baseline_mae > 0 else float("nan")
        rows.append({"split_name": split_name, "best_learned_model": best["model_name"], "baseline_mae": baseline_mae, "best_learned_mae": float(best["mae"]), "relative_mae_improvement": improvement})
    comparison = pd.DataFrame(rows)
    improved_count = int((comparison["relative_mae_improvement"] > 0.02).sum()) if not comparison.empty else 0
    if not bool(validation["success"].all()):
        return "constraining/contradictory", improved_count, comparison
    if improved_count >= 2:
        return "supportive", improved_count, comparison
    return "null", improved_count, comparison


def build_reports(
    *,
    full_report_path: Path,
    artifacts_written: Sequence[str],
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    split_summaries: pd.DataFrame,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
    outcome: str,
    feature_stats: Mapping[str, Any],
    unit_test_result: Mapping[str, Any] | None,
    model_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    validation_success = bool(validation["success"].all())
    best_rows = metrics.sort_values(["split_name", "mae"]).groupby("split_name").head(1)
    neural_devices = sorted(
        {
            str(record.get("device"))
            for record in model_records
            if record.get("modelName") == "neural_embedding_mlp" and record.get("device")
        }
    )
    neural_device_text = ", ".join(neural_devices) if neural_devices else "not recorded"
    caveats = (
        "Targets are heterogeneous oriented signed-log metric values; the modeling dataset is a deterministic balanced sample from the full S04 corpus; "
        "held-out rows with blank policy/world/goal/perturbation keys are excluded from the corresponding group split; source imbalance and proxy goal links remain limiting factors."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    text = f"""# E07 S05 Full Results: Behavior Predictor

## Top Summary

- Research step ID: S05
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S05 trained leakage-aware behavior-prediction baselines and a compact neural surrogate on a balanced sample of the S04 corpus, then tested whether predictions generalize to held-out policies, worlds, goals, and perturbation labels.
- Recommended next action: Chief review of the split metrics and source-imbalance limits; proceed to S06 only after deciding whether these S05 surrogate outputs are adequate for policy embeddings.

## Frozen Question

Can a GPU-trained surrogate predict competence vectors from policy representation, world representation, goal, and perturbation regime?

## Inputs

- S04 corpus: `{args.s04_corpus}` (SHA-256 `{feature_stats['s04CorpusSha256']}`)
- S01 world inventory: `{args.s01_world_inventory}`
- S02 policy representation table: `{args.s02_policy_table}`
- S03 goal representation table: `{args.s03_goal_table}`

## Methods

The script read the full S04 corpus to record coverage and source balance, then created a deterministic balanced modeling set capped at {args.max_rows_per_source_metric} rows per `(source_experiment_id, source_table, source_metric_name)` stratum. This prevents the largest E02 and E06 tables from dominating every fit while preserving all source experiments and metric families.

The regression target is an oriented signed-log metric value: metrics marked `minimize` are sign-flipped and all values are transformed by `sign(x) * log1p(abs(x))`. This keeps count-like and percent-like metrics in one bounded proxy target while preserving target direction.

Leakage-safe splits were defined by deterministic group holdout over `canonical_policy_id`, `world_id`, `canonical_goal_id`, and `perturbation_type`. Rows with blank keys for a split were excluded from that split's train/test evaluation and counted in the split summary. Baselines were evaluated first: a global median and a source-metric median. Then a hashed linear SGD surrogate and a PyTorch embedding MLP were trained and evaluated on the same splits.

Calibration was checked by regression slope/intercept of observed target on prediction, Pearson correlation, and decile-bin predicted-vs-observed absolute gaps.

## Commands

- `python -m unittest tests.e07.test_predictor_schema`
- `python scripts/e07_s05_behavior_predictor.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- scikit-learn SGDRegressor and FeatureHasher
- PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}
- Worker count: PyTorch CPU thread cap set to {min(8, os.cpu_count() or 1)}; neural model device(s): `{neural_device_text}`.
- Repository commit before S05 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Results

Full S04 rows: {feature_stats['s04CorpusRows']}
Balanced modeling rows: {feature_stats['modelingRows']}

Rows by source in full S04 corpus:

{markdown_table(pd.DataFrame([{'source_experiment_id': key, 'row_count': value} for key, value in feature_stats['rowsBySourceExperiment'].items()]))}

Rows by source in modeling sample:

{markdown_table(pd.DataFrame([{'source_experiment_id': key, 'row_count': value} for key, value in feature_stats['modelingRowsBySourceExperiment'].items()]))}

### Split Summary

{markdown_table(split_summaries)}

### Predictor Metrics

{markdown_table(metrics[['split_name', 'model_name', 'model_kind', 'train_rows', 'test_rows', 'mae', 'rmse', 'r2', 'calibration_slope', 'calibration_ece']].sort_values(['split_name', 'mae']))}

### Best Model by Split

{markdown_table(best_rows[['split_name', 'model_name', 'mae', 'rmse', 'r2', 'calibration_slope', 'calibration_ece']])}

### Learned Model Comparison Against Source-Metric Median

{markdown_table(comparison)}

## Validation

{markdown_table(validation)}

## Calibration

Calibration bins are written to the calibration artifact. Summary rows:

{markdown_table(calibration.groupby(['split_name', 'model_name']).agg(weighted_abs_gap=('abs_gap', 'mean'), bin_count=('bin_index', 'count')).reset_index())}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- The result is a computational surrogate over previously generated metrics, not direct biological or causal validation.
- The target combines heterogeneous metrics through a signed-log transform; it is useful for relative prediction stress tests but not a physical unit.
- S04 source imbalance remains substantial even after deterministic balancing, especially E02 null metrics and E06 prediction/governance rows.
- Group holdouts exclude rows lacking the split key, so policy/world/goal/perturbation generalization is evaluated only where upstream keys were available.
- Proxy goal links from S04 are treated as features, not definitive ground-truth objectives.
- The neural model is intentionally small; S05 does not claim final architecture optimality.

## Recommended Next Action

Chief review should inspect whether the learned-model improvements, calibration, and leakage-safe split behavior are adequate for S06 policy embeddings. Do not start S06 until explicitly instructed.
"""
    write_text(full_report_path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e07"
    model_dir = artifacts_dir / "models" / "e07_behavior_predictor"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_predictor_schema"], args.repo_dir)

    frame, feature_stats = load_feature_frame(args)
    modeling_dataset_path = results_dir / "e07_behavior_predictor_modeling_dataset.parquet"
    frame.to_parquet(modeling_dataset_path, index=False)

    metrics_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for split_name, group_column in SPLIT_DEFINITIONS.items():
        print(f"[S05] evaluating {split_name} by {group_column}", file=sys.stderr, flush=True)
        split_metrics, split_calibration, split_summary_row, split_models = evaluate_split(
            frame,
            split_name=split_name,
            group_column=group_column,
            args=args,
            model_dir=model_dir,
        )
        metrics_rows.extend(split_metrics)
        calibration_frames.extend(split_calibration)
        split_rows.append(split_summary_row)
        model_records.extend(split_models)

    metrics = pd.DataFrame(metrics_rows)
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    split_summaries = pd.DataFrame(split_rows)
    validation = validate_prediction_artifacts(metrics, split_summaries)
    outcome, improved_count, comparison = result_classification(metrics, validation)

    metrics_path = results_dir / "e07_behavior_predictor_metrics.parquet"
    metrics_csv_path = tables_dir / "e07_behavior_predictor_metrics.csv"
    calibration_path = results_dir / "e07_behavior_predictor_calibration.parquet"
    calibration_csv_path = tables_dir / "e07_behavior_predictor_calibration.csv"
    split_summary_path = tables_dir / "e07_behavior_predictor_split_summary.csv"
    comparison_path = tables_dir / "e07_behavior_predictor_baseline_comparison.csv"
    feature_manifest_path = results_dir / "e07_behavior_predictor_feature_manifest.json"
    validation_path = step_dir / "e07_s05_validation_checks.csv"
    figure_path = figures_dir / "behavior_predictor_performance.png"
    full_report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    model_manifest_path = model_dir / "model_manifest.json"
    config_path = model_dir / "s05_config.json"

    metrics.to_parquet(metrics_path, index=False)
    metrics.to_csv(metrics_csv_path, index=False)
    calibration.to_parquet(calibration_path, index=False)
    calibration.to_csv(calibration_csv_path, index=False)
    split_summaries.to_csv(split_summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    validation.to_csv(validation_path, index=False)
    plot_performance(metrics, figure_path)

    feature_manifest = {
        "schemaVersion": PREDICTOR_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "createdAt": utc_now(),
        "categoricalColumns": list(CATEGORICAL_COLUMNS),
        "numericColumns": list(NUMERIC_COLUMNS),
        "targetTransform": "sign(oriented_metric_value) * log1p(abs(oriented_metric_value)); minimize metrics are sign-flipped",
        "splitDefinitions": SPLIT_DEFINITIONS,
        "featureStats": feature_stats,
    }
    write_json(feature_manifest_path, feature_manifest)
    write_json(
        config_path,
        {
            "schemaVersion": PREDICTOR_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "maxRowsPerSourceMetric": args.max_rows_per_source_metric,
            "maxLinearTrainRows": args.max_linear_train_rows,
            "maxNeuralTrainRows": args.max_neural_train_rows,
            "neuralEpochs": args.neural_epochs,
            "batchSize": args.batch_size,
            "torchCudaAvailable": torch.cuda.is_available(),
            "torchThreads": torch.get_num_threads(),
        },
    )
    write_json(
        model_manifest_path,
        {
            "schemaVersion": PREDICTOR_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "createdAt": utc_now(),
            "outcomeClassification": outcome,
            "learnedModelImprovedSplitCount": improved_count,
            "modelRecords": model_records,
            "metricsPath": str(metrics_path),
            "calibrationPath": str(calibration_path),
            "featureManifestPath": str(feature_manifest_path),
        },
    )

    artifacts_written = [
        str(full_report_path),
        str(model_dir),
        str(model_manifest_path),
        str(config_path),
        str(metrics_path),
        str(metrics_csv_path),
        str(calibration_path),
        str(calibration_csv_path),
        str(split_summary_path),
        str(comparison_path),
        str(modeling_dataset_path),
        str(feature_manifest_path),
        str(figure_path),
        str(validation_path),
        str(artifact_manifest_path),
    ]
    build_reports(
        full_report_path=full_report_path,
        artifacts_written=artifacts_written,
        metrics=metrics,
        calibration=calibration,
        split_summaries=split_summaries,
        validation=validation,
        comparison=comparison,
        outcome=outcome,
        feature_stats=feature_stats,
        unit_test_result=unit_test_result,
        model_records=model_records,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s05.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": bool(validation["success"].all()) and (unit_test_result is None or bool(unit_test_result["success"])),
        "outcomeClassification": outcome,
        "validationResult": {
            "passed": int(validation["success"].sum()),
            "total": int(len(validation)),
            "allPassed": bool(validation["success"].all()),
        },
        "unitTestResult": unit_test_result,
        "artifacts": [
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S05 full-results report."),
            artifact_entry(model_dir, artifacts_dir, "S05 trained model directory."),
            artifact_entry(model_manifest_path, artifacts_dir, "Model manifest."),
            artifact_entry(config_path, artifacts_dir, "S05 model/training configuration."),
            artifact_entry(metrics_path, artifacts_dir, "Primary predictor metrics in Parquet format."),
            artifact_entry(metrics_csv_path, artifacts_dir, "Primary predictor metrics in CSV format."),
            artifact_entry(calibration_path, artifacts_dir, "Calibration bins in Parquet format."),
            artifact_entry(calibration_csv_path, artifacts_dir, "Calibration bins in CSV format."),
            artifact_entry(split_summary_path, artifacts_dir, "Leakage-safe split summary."),
            artifact_entry(comparison_path, artifacts_dir, "Learned-model comparison against source-metric median baseline."),
            artifact_entry(modeling_dataset_path, artifacts_dir, "Balanced modeling dataset derived from S04."),
            artifact_entry(feature_manifest_path, artifacts_dir, "Feature and target transform manifest."),
            artifact_entry(figure_path, artifacts_dir, "Behavior predictor performance figure."),
            artifact_entry(validation_path, artifacts_dir, "S05 validation checks."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest."),
        ],
    }
    write_json(artifact_manifest_path, manifest_payload)
    manifest_payload["artifacts"][-1]["sizeBytes"] = artifact_manifest_path.stat().st_size
    write_json(artifact_manifest_path, manifest_payload)


if __name__ == "__main__":
    main()
