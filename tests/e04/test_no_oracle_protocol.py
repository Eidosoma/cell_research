"""E04 S07 no-global-oracle protocol tests."""

from __future__ import annotations

import json
import unittest

from src.e03.policy_interface import PolicyObservation
from src.e04.memory_policies import CellMemoryState
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    CENTRALIZED_BASELINE_TYPE,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    audit_training_feature_dict,
    centralized_baseline_contract,
    legacy_policy_observation_audit,
    project_local_training_observation,
)
from src.e04.signaling import LocalSignalValues, SignalSensation


class NoOracleProtocolTests(unittest.TestCase):
    def _observation(self) -> PolicyObservation:
        return PolicyObservation(
            actor_index=2,
            actor_thread_id=12,
            actor_value=5,
            actor_label="bubble",
            actor_status="ACTIVE",
            behavior="bubble",
            values=(1, 8, 5, 4, 9),
            labels=("bubble", "bubble", "bubble", "bubble", "bubble"),
            statuses=("ACTIVE", "ACTIVE", "ACTIVE", "FREEZE", "ACTIVE"),
            left_boundary=0,
            right_boundary=4,
            reverse_direction=False,
            ideal_position=4,
            group_status="ACTIVE",
        )

    def _sensation(self) -> SignalSensation:
        return SignalSensation(
            event_step=7,
            actor_thread_id=12,
            actor_position=2,
            local_signals=((2, LocalSignalValues(blocked=0.1, sorted=1.0, target_seeking=1.0)),),
            diffusive_fields={
                "blocked": 0.25,
                "frustrated": 0.5,
                "sorted": 1.0,
                "target_seeking": 1.0,
                "morphogen": 1.0,
            },
            accessed_indices=(1, 2, 3),
            access_scope="local_window_plus_explicit_diffusive_fields",
            noise_applied=False,
        )

    def test_projected_training_view_excludes_oracle_fields(self) -> None:
        view = project_local_training_observation(
            self._observation(),
            CellMemoryState(last_move_success=False, time_since_movement=3, local_frustration=4),
            self._sensation(),
        )
        features = view.to_feature_dict()
        audit = audit_training_feature_dict(features)
        serialized = json.dumps(features, sort_keys=True)

        self.assertFalse(audit["usesGlobalOracle"])
        self.assertNotIn("ideal_position", serialized)
        self.assertNotIn("target_position", serialized)
        self.assertNotIn("sortedness", serialized)
        self.assertNotIn("signal_target_seeking", serialized)
        self.assertNotIn("signal_morphogen", serialized)
        self.assertEqual(features["left_value"], 8)
        self.assertEqual(features["right_status"], "FREEZE")
        self.assertEqual(features["signal_blocked"], 0.25)
        self.assertEqual(features["signal_frustrated"], 0.5)

    def test_legacy_policy_observation_exposure_is_documented(self) -> None:
        audit = legacy_policy_observation_audit()

        self.assertTrue(audit["requiresProjectionForLocalOnlyTraining"])
        self.assertIn("values", audit["oracleExposureFields"])
        self.assertIn("ideal_position", audit["oracleExposureFields"])

    def test_signal_allowlist_excludes_target_fields(self) -> None:
        self.assertEqual(set(ALLOWED_TRAINING_SIGNAL_FIELDS), {"blocked", "frustrated"})
        self.assertIn("target_seeking", EXCLUDED_TRAINING_SIGNAL_FIELDS)
        self.assertIn("morphogen", EXCLUDED_TRAINING_SIGNAL_FIELDS)

    def test_centralized_baseline_contract_is_explicitly_ineligible(self) -> None:
        contract = centralized_baseline_contract()

        self.assertEqual(contract["baselineType"], CENTRALIZED_BASELINE_TYPE)
        self.assertTrue(contract["usesGlobalOracle"])
        self.assertFalse(contract["eligibleForLocalClaims"])
        self.assertTrue(contract["mustNotBeMixedWithLocalOnlyRows"])


if __name__ == "__main__":
    unittest.main()
