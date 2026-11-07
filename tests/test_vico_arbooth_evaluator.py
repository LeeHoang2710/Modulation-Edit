"""Static CPU-portability contracts for the ViCo evaluator."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SimilarityTests(unittest.TestCase):
    def test_cpu_similarity_casts_half_features_to_float32(self):
        source = (ROOT / "benchmark/evaluate_vico_arbooth_protocol.py").read_text()
        self.assertIn("left.float() @ right.float().T", source)
        self.assertIn("reference_features.float() @ generated_features.float().T", source)
        self.assertGreaterEqual(source.count("outputs.append(features.float().cpu())"), 2)


if __name__ == "__main__":
    unittest.main()
