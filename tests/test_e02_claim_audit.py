import unittest
from pathlib import Path

import pandas as pd

from scripts import e02_s15_claim_audit as s15


def synthetic_e01_claims() -> pd.DataFrame:
    rows = []
    for claim_id in sorted(s15.CLAIM_VERDICTS):
        rows.append(
            {
                "claimId": claim_id,
                "paperFigure": claim_id.split("_", 1)[0],
                "claimType": "synthetic",
                "paperClaim": f"Claim {claim_id}",
                "paperExpectedValue": "",
                "classification": "directionally consistent",
                "observedEvidence": "synthetic evidence",
                "justification": "synthetic justification",
                "reviewPriority": "normal",
            }
        )
    return pd.DataFrame(rows)


class ClaimAuditTests(unittest.TestCase):
    def test_claim_verdicts_are_allowed_and_complete(self):
        claims = synthetic_e01_claims()
        table = s15.build_claim_audit_rows(claims)
        self.assertEqual(len(table), 32)
        self.assertEqual(set(table["claimId"]), set(claims["claimId"]))
        self.assertTrue(set(table["e02Verdict"]).issubset(s15.ALLOWED_VERDICTS))
        self.assertTrue(table["claimId"].is_unique)
        self.assertTrue(table["verdictRationale"].str.len().gt(0).all())

    def test_claim_map_mismatch_raises(self):
        claims = synthetic_e01_claims().iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "Claim verdict map mismatch"):
            s15.build_claim_audit_rows(claims)

    def test_alternative_matrix_has_every_mechanism_per_claim(self):
        claims = synthetic_e01_claims()
        table = s15.build_claim_audit_rows(claims)
        matrix = s15.build_alternative_matrix(table)
        self.assertEqual(len(matrix), len(table) * len(s15.MECHANISMS))
        per_claim = matrix.groupby("claimId")["mechanismId"].nunique()
        self.assertTrue(per_claim.eq(len(s15.MECHANISMS)).all())
        self.assertIn("S08_dg_matched_nulls", set(matrix["mechanismId"]))
        dg = matrix[(matrix["claimId"] == "FIG07_DG_ALGORITHM_ORDERING") & (matrix["mechanismId"] == "S08_dg_matched_nulls")]
        self.assertEqual(dg.iloc[0]["assessment"], "explains or challenges")

    def test_validate_outputs_flags_missing_referenced_artifact(self):
        claims = synthetic_e01_claims()
        table = s15.build_claim_audit_rows(claims)
        matrix = s15.build_alternative_matrix(table)
        table.loc[0, "primaryEvidenceArtifacts"] = "/definitely/missing/artifact.parquet"
        status_index = pd.DataFrame(
            {
                "sourceResearchStepId": [f"S{i:02d}" for i in range(1, 15)],
                "success": [True] * 14,
                "validationResult": ["passed"] * 14,
                "statusPath": [str(Path("/tmp") / f"S{i:02d}.json") for i in range(1, 15)],
            }
        )
        validation = s15.validate_outputs(
            claims,
            table,
            matrix,
            status_index,
            required_artifacts=[],
        )
        row = validation[validation["checkId"] == "referenced_artifacts_exist"].iloc[0]
        self.assertFalse(bool(row["validationPassed"]))
        self.assertIn("missingCount", row["detail"])


if __name__ == "__main__":
    unittest.main()
