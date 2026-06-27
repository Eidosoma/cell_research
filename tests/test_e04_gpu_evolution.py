from __future__ import annotations

import unittest

import numpy as np

from memory_repair import (
    EVOLUTION_SEARCH_VERSION,
    GENOME_WIDTH,
    EvolutionSearchConfig,
    audit_candidate_record,
    candidate_record_from_genome,
    cpu_gpu_agreement_smoke,
    policy_from_genome,
    proxy_fitness_numpy,
    proxy_fitness_torch,
    replay_candidate,
    replay_fingerprint,
    run_gpu_evolution_search,
    seeded_reference_genome,
    summarize_replay_advantage,
)
from memory_repair.training_constraints import audit_policy_spec_for_oracle_access, local_only_training_protocol


class TestE04GpuEvolution(unittest.TestCase):
    def test_cpu_torch_proxy_agreement(self) -> None:
        genomes = np.vstack([seeded_reference_genome(), np.linspace(0.05, 0.95, GENOME_WIDTH)])
        cpu_scores = proxy_fitness_numpy(genomes)
        torch_scores = proxy_fitness_torch(genomes, device="cpu").detach().cpu().numpy()
        self.assertTrue(np.allclose(cpu_scores, torch_scores, atol=1e-12))
        smoke = cpu_gpu_agreement_smoke(seed=808, device="cpu")
        self.assertTrue(smoke["success"], smoke)
        self.assertEqual(smoke["evolutionSearchVersion"], EVOLUTION_SEARCH_VERSION)

    def test_candidate_policy_passes_s07_local_only_audit(self) -> None:
        candidate = candidate_record_from_genome(seeded_reference_genome(), seed=808)
        audit = audit_candidate_record(candidate)
        self.assertTrue(audit["success"], audit)
        policy = policy_from_genome(seeded_reference_genome(), seed=808)
        self.assertTrue(audit_policy_spec_for_oracle_access(policy, local_only_training_protocol())["success"])

    def test_small_evolution_search_returns_audited_elites(self) -> None:
        result = run_gpu_evolution_search(
            EvolutionSearchConfig(population_size=8, generations=2, elite_count=2, mutation_scale=0.05, seed=808, device="cpu")
        )
        self.assertEqual(len(result["trainingCurveRows"]), 2)
        self.assertEqual(len(result["eliteCandidates"]), 2)
        self.assertTrue(all(candidate["audit"]["success"] for candidate in result["eliteCandidates"]))
        best_scores = [row["bestProxyFitness"] for row in result["trainingCurveRows"]]
        self.assertGreaterEqual(best_scores[-1], best_scores[0] - 1e-12)

    def test_replay_is_stable_and_beats_fixed_on_one_repair_case(self) -> None:
        result = run_gpu_evolution_search(
            EvolutionSearchConfig(population_size=8, generations=2, elite_count=2, mutation_scale=0.05, seed=808, device="cpu")
        )
        best = result["eliteCandidates"][0]
        first = replay_candidate(best, base_seed=9000, split="unit", repair_horizon=24)
        second = replay_candidate(best, base_seed=9000, split="unit", repair_horizon=24)
        self.assertEqual(replay_fingerprint(first["replayRows"]), replay_fingerprint(second["replayRows"]))
        advantage = summarize_replay_advantage(first["replayRows"])
        self.assertTrue(advantage["success"], advantage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
