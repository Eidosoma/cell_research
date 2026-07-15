#!/usr/bin/env python3
"""Create compact cost and cross-metric summaries from the exact S07 tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


S04 = Path("/artifacts/research_steps/S04")
OUTPUT = Path("/artifacts/research_steps/S07")
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
LABEL_COLUMNS = [
    "shortest_activations",
    "minimum_observation_reads",
    "minimum_observation_reads_activations",
    "minimum_value_comparisons",
    "minimum_value_comparisons_activations",
    "minimum_accepted_swaps",
    "minimum_accepted_swaps_activations",
    "minimum_displaced_cells",
    "minimum_full_ledger_activations",
    "minimum_full_ledger_observation_reads",
    "minimum_full_ledger_value_comparisons",
    "minimum_full_ledger_no_ops",
    "minimum_full_ledger_rejections",
    "minimum_full_ledger_memory_updates",
    "minimum_full_ledger_accepted_swaps",
    "minimum_full_ledger_displaced_cells",
]


def row_group_map(path: Path) -> tuple[pq.ParquetFile, dict[int, list[int]]]:
    parquet = pq.ParquetFile(path)
    column = parquet.schema_arrow.names.index("family_ordinal")
    result: dict[int, list[int]] = {}
    for row_group in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(row_group).column(column).statistics
        if statistics is None or int(statistics.min) != int(statistics.max):
            raise AssertionError("S07 output row group spans multiple families")
        result.setdefault(int(statistics.min), []).append(row_group)
    return parquet, result


def read_family(
    parquet: pq.ParquetFile, groups: dict[int, list[int]], ordinal: int
) -> pd.DataFrame:
    return parquet.read_row_groups(groups[ordinal]).to_pandas()


def family_cost_summary(
    ordinal: int, metadata: pd.Series, cost: pd.DataFrame
) -> dict[str, Any]:
    reachable_active = (cost.terminal_code == 0) & (cost.shortest_activations >= 0)
    result: dict[str, Any] = {
        "family_ordinal": ordinal,
        "materialization_tier": metadata.materialization_tier,
        "n": int(metadata.n),
        "architecture": metadata.architecture,
        "direction": metadata.direction,
        "policy_code": int(metadata.policy_code),
        "policy_profile": metadata.policy_profile,
        "selection_owner_count": int(metadata.selection_owner_count),
        "fault_code": int(metadata.fault_code),
        "fault_mode": metadata.fault_mode,
        "fault_count": int(metadata.fault_count),
        "fault_identity_ids": ",".join(metadata.fault_identity_ids),
        "state_count": len(cost),
        "complete_start_count": int((cost.terminal_code == 1).sum()),
        "reachable_active_count": int(reachable_active.sum()),
        "unreachable_active_count": int(
            ((cost.terminal_code == 0) & (cost.shortest_activations < 0)).sum()
        ),
        "quiescent_count": int((cost.terminal_code == 2).sum()),
    }
    for column in LABEL_COLUMNS:
        values = cost.loc[reachable_active, column].to_numpy(dtype=np.int64)
        result[f"{column}_sum"] = int(values.sum()) if len(values) else 0
        result[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
        result[f"{column}_median"] = float(np.median(values)) if len(values) else float("nan")
        result[f"{column}_p90"] = float(np.quantile(values, 0.9)) if len(values) else float("nan")
        result[f"{column}_maximum"] = int(values.max()) if len(values) else -1
    return result


def aggregate_cost_strata(frame: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("overall", []),
        ("n", ["n"]),
        ("architecture", ["architecture"]),
        ("direction", ["direction"]),
        ("policy_profile", ["policy_profile"]),
        ("fault_mode", ["fault_mode"]),
        ("fault_count", ["fault_count"]),
        (
            "planned_regime",
            [
                "n",
                "architecture",
                "direction",
                "policy_profile",
                "fault_mode",
                "fault_count",
            ],
        ),
    ]
    count_columns = [
        "state_count",
        "complete_start_count",
        "reachable_active_count",
        "unreachable_active_count",
        "quiescent_count",
    ]
    rows: list[dict[str, Any]] = []
    for stratum_type, columns in specifications:
        grouped = (
            frame.groupby(columns, dropna=False, sort=True)
            if columns
            else [((), frame)]
        )
        for key, group in grouped:
            values = key if isinstance(key, tuple) else (key,)
            reachable = int(group.reachable_active_count.sum())
            row: dict[str, Any] = {
                "stratum_type": stratum_type,
                "stratum_value": (
                    "all"
                    if not columns
                    else "|".join(
                        f"{name}={value}"
                        for name, value in zip(columns, values, strict=True)
                    )
                ),
                "family_count": len(group),
                **{column: int(group[column].sum()) for column in count_columns},
            }
            for label in LABEL_COLUMNS:
                total = int(group[f"{label}_sum"].sum())
                row[f"{label}_sum"] = total
                row[f"{label}_mean"] = (
                    float(total / reachable) if reachable else float("nan")
                )
                row[f"{label}_maximum"] = int(group[f"{label}_maximum"].max())
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("overall", []),
        ("n", ["n"]),
        ("architecture", ["architecture"]),
        ("direction", ["direction"]),
        ("policy_profile", ["policy_profile"]),
        ("fault_mode", ["fault_mode"]),
    ]
    rows: list[dict[str, Any]] = []
    for stratum_type, columns in specifications:
        keys = ["scope", "signature_code", *columns]
        grouped = frame.groupby(keys, dropna=False, sort=True)
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            scope, signature = key_values[:2]
            values = key_values[2:]
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": (
                        "all"
                        if not columns
                        else "|".join(
                            f"{name}={value}"
                            for name, value in zip(columns, values, strict=True)
                        )
                    ),
                    "scope": scope,
                    "signature_code": int(signature),
                    "necessary_metrics": ";".join(
                        metric
                        for metric, code in METRIC_CODES.items()
                        if int(signature) & (1 << code)
                    )
                    or "none",
                    "family_count_contributing": int((group.state_count > 0).sum()),
                    "state_count": int(group.state_count.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    metadata = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    cost_pf, cost_groups = row_group_map(OUTPUT / "minimum_cost_solutions.parquet")
    path_pf, path_groups = row_group_map(OUTPUT / "path_solutions.parquet")
    cost_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for position, ordinal in enumerate(sorted(cost_groups)):
        cost = read_family(cost_pf, cost_groups, ordinal).sort_values("state_ordinal")
        path = read_family(path_pf, path_groups, ordinal).sort_values(
            ["metric_code", "state_ordinal"]
        )
        cost_rows.append(family_cost_summary(ordinal, metadata.loc[ordinal], cost))
        classes = np.vstack(
            [
                path[path.metric_code == code]
                .sort_values("state_ordinal")
                .classification_code.to_numpy()
                for code in range(len(METRICS))
            ]
        )
        signature = np.zeros(len(cost), dtype=np.uint8)
        for code in range(len(METRICS)):
            signature |= ((classes[code] == 2).astype(np.uint8) << code)
        reachable_active = (classes[0] == 1) | (classes[0] == 2)
        for scope, mask in (
            ("all_states", np.ones(len(cost), dtype=bool)),
            ("reachable_active_only", reachable_active),
        ):
            counts = np.bincount(signature[mask], minlength=16)
            for code, count in enumerate(counts):
                if count:
                    signature_rows.append(
                        {
                            "family_ordinal": ordinal,
                            "n": int(metadata.loc[ordinal].n),
                            "architecture": metadata.loc[ordinal].architecture,
                            "direction": metadata.loc[ordinal].direction,
                            "policy_profile": metadata.loc[ordinal].policy_profile,
                            "fault_mode": metadata.loc[ordinal].fault_mode,
                            "scope": scope,
                            "signature_code": code,
                            "state_count": int(count),
                        }
                    )
        if position % 500 == 0:
            print(json.dumps({"familyPosition": position, "familyOrdinal": ordinal}), flush=True)
    family_cost = pd.DataFrame(cost_rows)
    stratum_cost = aggregate_cost_strata(family_cost)
    signature_family = pd.DataFrame(signature_rows)
    signature_summary = aggregate_signatures(signature_family)
    for frame, name in (
        (family_cost, "cost_solution_summary_by_family.parquet"),
        (stratum_cost, "cost_solution_summary_by_stratum.parquet"),
        (signature_family, "necessity_signature_counts_by_family.parquet"),
        (signature_summary, "necessity_signature_counts.parquet"),
    ):
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            OUTPUT / name,
            compression="zstd",
            compression_level=7,
        )
    result = {
        "schemaVersion": "e03.s07.compact_summary.v1",
        "researchStepId": "S07",
        "success": True,
        "familyCostRows": len(family_cost),
        "stratumCostRows": len(stratum_cost),
        "familySignatureRows": len(signature_family),
        "signatureSummaryRows": len(signature_summary),
    }
    (OUTPUT / "compact_summary.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    prevalence = pd.read_parquet(OUTPUT / "necessity_prevalence_by_family.parquet")
    overall_reachable = signature_summary[
        (signature_summary.stratum_type == "overall")
        & (signature_summary.scope == "reachable_active_only")
    ]
    overall_all = signature_summary[
        (signature_summary.stratum_type == "overall")
        & (signature_summary.scope == "all_states")
    ]
    metric_marginals = {}
    for metric, code in METRIC_CODES.items():
        observed = int(
            overall_reachable.loc[
                (overall_reachable.signature_code.astype(int) & (1 << code)) != 0,
                "state_count",
            ].sum()
        )
        expected = int(
            prevalence.loc[prevalence.metric == metric, "necessary_detour_count"].sum()
        )
        metric_marginals[metric] = {
            "signatureCount": observed,
            "prevalenceCount": expected,
            "matched": observed == expected,
        }
    validation = {
        "schemaVersion": "e03.s07.summary_validation.v1",
        "researchStepId": "S07",
        "success": (
            len(family_cost) == 7984
            and int(overall_reachable.state_count.sum()) == 9214057
            and int(overall_all.state_count.sum()) == 22301808
            and all(row["matched"] for row in metric_marginals.values())
        ),
        "familyCostRows": len(family_cost),
        "reachableActiveSignatureCount": int(overall_reachable.state_count.sum()),
        "allStateSignatureCount": int(overall_all.state_count.sum()),
        "metricMarginals": metric_marginals,
    }
    (OUTPUT / "summary_validation.json").write_text(
        json.dumps(validation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if not validation["success"]:
        raise AssertionError(validation)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
