"""Contracts for the self-contained two-subject Vast.ai experiment."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references" / "vico_16"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_COUNTS = {
    "alarm_clock": 5,
    "black_cat": 5,
    "brown_cat": 5,
    "brown_dog": 5,
    "brown_dog2": 5,
    "brown_teddybear": 7,
    "cat_statue": 6,
    "duck_toy": 4,
    "elephant_statue": 5,
    "fat_dog": 6,
    "monster_toy": 5,
    "pink_sunglasses": 6,
    "red_teapot": 5,
    "red_vase": 6,
    "tortoise_plushy": 12,
    "wooden_barn": 7,
}


class SpatialExperimentAssetTests(unittest.TestCase):
    def test_all_16_first_party_reference_sets_are_present(self):
        actual = {}
        for directory in REFERENCES.iterdir():
            if directory.is_dir():
                actual[directory.name] = sum(
                    path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir()
                )
        self.assertEqual(actual, EXPECTED_COUNTS)
        self.assertEqual(sum(actual.values()), 94)

    def test_runner_has_no_comparison_method_dependency(self):
        runner = (ROOT / "scripts" / "run_fat_dog_grey_cat_spatial_test.sh").read_text()
        self.assertIn("references/vico_16", runner)
        self.assertNotIn("models/DreamMatcher", runner)
        self.assertNotIn("setup_vico_vastai", runner)

    def test_runner_uses_seed_three_and_paper_main_scales(self):
        runner = (ROOT / "scripts" / "run_fat_dog_grey_cat_spatial_test.sh").read_text()
        self.assertIn("TRAIN_SEED=3", runner)
        self.assertIn("INFER_SEED=3", runner)
        self.assertIn("--loss_scales 0,2-5", runner)
        self.assertIn("--offset-scales 0,2-5", runner)
        self.assertIn("seed_003.png", runner)

    def test_vast_guide_does_not_initialize_submodules(self):
        guide = (ROOT / "MULTI_SUBJECT_SPATIAL_MODULATION.md").read_text()
        self.assertNotIn("--recurse-submodules", guide)
        self.assertNotIn("git submodule update", guide)
        self.assertIn("bash scripts/setup_vastai_env.sh", guide)

    def test_blackwell_setup_cannot_reinstall_legacy_torch(self):
        setup = (ROOT / "scripts" / "setup_vastai_env.sh").read_text()
        runner = (ROOT / "scripts" / "run_fat_dog_grey_cat_spatial_test.sh").read_text()
        self.assertIn('TORCH_VERSION="${TORCH_VERSION:-2.9.1}"', setup)
        self.assertIn("https://download.pytorch.org/whl/cu128", setup)
        self.assertIn("FILTERED_REQUIREMENTS", setup)
        self.assertIn("prebuilt FlashAttention", setup)
        self.assertNotIn("--no-build-isolation --no-deps", setup)
        self.assertIn("repair_rtx5090_env.sh", runner)


if __name__ == "__main__":
    unittest.main()
