from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from causal_simulator.pairing import (
    DIRECTIONS,
    FAULT_PROFILES,
    PAIRING_PRESPECIFICATION_SHA256,
    POLICIES,
    STREAM_SPECIFICATION_SHA256,
    algotype_assignment_id,
    assign_fault_profiles,
    contrast_catalog,
    coupling_matrix_rows,
    pairing_block_content,
    pairing_block_id,
    scenario_variant_id,
    sha256_file,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def mock_record(index: int, *, split: str = "screening_pool") -> dict:
    return {
        "inputScenarioId": f"i1:{index:064x}",
        "split": split,
        "n": 20,
        "valueProfile": "unique",
        "orderStructure": "nearly_sorted",
        "replicateOrdinal": index,
        "initialValuesSha256": f"{index + 1:064x}",
        "initialOccupancySha256": f"{index + 2:064x}",
        "scenarioSeed": str(index + 100),
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "eventBudgetOpportunities": 40_000,
    }


class PairingContractTests(unittest.TestCase):
    def test_specs_were_frozen_and_hash_locked(self) -> None:
        pairing = REPOSITORY / "design/s08/pairing_prespecification.json"
        streams = REPOSITORY / "design/s08/semantic_random_stream_specification.json"
        self.assertEqual(sha256_file(pairing), PAIRING_PRESPECIFICATION_SHA256)
        self.assertEqual(sha256_file(streams), STREAM_SPECIFICATION_SHA256)
        self.assertTrue(json.loads(pairing.read_text())["frozenBeforeImplementation"])
        self.assertTrue(json.loads(streams.read_text())["frozenBeforeImplementation"])

    def test_fault_profile_allocation_is_deterministic_and_near_balanced(self) -> None:
        records = [mock_record(index) for index in range(250)]
        first = assign_fault_profiles(records)
        second = assign_fault_profiles(list(reversed(records)))
        self.assertEqual(first, second)
        counts = Counter(item.profile_id for item in first.values())
        self.assertEqual(len(counts), len(FAULT_PROFILES))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(sum(counts.values()), 250)

    def test_pairing_blocks_cross_three_policies_and_two_directions(self) -> None:
        record = mock_record(1)
        ids = set()
        for policy in POLICIES:
            assignment = algotype_assignment_id(20, policy)
            for direction in DIRECTIONS:
                content = pairing_block_content(
                    record,
                    policy=policy,
                    direction=direction,
                    algotype_id=assignment,
                    fault_map_id="fp1:" + "a" * 64,
                )
                ids.add(pairing_block_id(content))
        self.assertEqual(len(ids), 6)

    def test_scenario_variants_share_block_but_not_identity(self) -> None:
        block = "pb1:" + "b" * 64
        fault_map = "fp1:" + "c" * 64
        variants = {
            scenario_variant_id(block, mobility, fault_map)
            for mobility in ("normal", "passive", "stuck")
        }
        self.assertEqual(len(variants), 3)

    def test_catalog_accounts_for_all_primary_estimands(self) -> None:
        catalog = contrast_catalog()
        self.assertEqual(catalog["primaryEstimandCount"], 12)
        self.assertEqual(catalog["executableEstimandCount"], 11)
        self.assertEqual(catalog["declaredUnpairedNonExecutableCount"], 1)
        self.assertEqual(catalog["executableArmCount"], 30)
        by_id = {item["estimandId"]: item for item in catalog["contrasts"]}
        self.assertEqual(by_id["E02-S01-E07"]["rngPairingStatus"], "not_executable")
        self.assertIn("inactive", by_id["E02-S01-E05"]["rngPairingStatus"])
        for estimand in ("E02-S01-E01", "E02-S01-E02"):
            settings = [arm["settings"] for arm in by_id[estimand]["arms"]]
            central = next(
                item for item in settings if item["architecture"] == "central_local_proposal_k1"
            )
            distributed = next(
                item for item in settings if item["architecture"] == "distributed_local"
            )
            self.assertEqual(central["coordinatorProfile"], "common_validator_only")
            self.assertEqual(distributed["coordinatorProfile"], "none")

    def test_incompatible_scheduler_streams_are_not_claimed_exact(self) -> None:
        rows = coupling_matrix_rows(contrast_catalog())
        scheduler = [
            item
            for item in rows
            if item["estimandId"] == "E02-S01-E03"
            and item["componentOrStream"] == "actor_selection"
        ]
        self.assertEqual(len(scheduler), 1)
        self.assertEqual(scheduler[0]["coupling"], "scenario_paired_rng_unpaired")


if __name__ == "__main__":
    unittest.main()
