"""Static, CPU-only contracts for Custom Diffusion and DreamMatcher adapters."""
import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name} in {path}")


class PersonalizationAdapterTests(unittest.TestCase):
    def test_dreammatcher_aliases_cover_released_source_images(self):
        adapter = ROOT / "benchmark/dreammatcher_vico_common.py"
        aliases = assignment_literal(adapter, "CONCEPT_INPUT_NAMES")
        source_map = json.loads((ROOT / "models/DreamMatcher/inputs/imgs_source.txt").read_text())
        self.assertEqual(len(aliases), 16)
        self.assertEqual(set(aliases.values()), set(source_map))
        for folder, filename in source_map.items():
            self.assertTrue((ROOT / "models/DreamMatcher/inputs" / folder / filename).is_file())

    def test_five_seed_defaults_and_distinct_outputs(self):
        for relative in (
            "benchmark/run_custom_diffusion_vico.py",
            "benchmark/run_dreammatcher_custom_diffusion_vico.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn('default="0,1,2,3,4"', source)
            self.assertIn('f"seed_{seed:03d}.png"', source)

    def test_custom_diffusion_uses_dreammatcher_compatible_pipeline(self):
        source = (ROOT / "benchmark/run_custom_diffusion_vico.py").read_text()
        self.assertIn("from dreammatcher.diffuser_utils import CustomDiffusionPipeline", source)
        self.assertIn("from diffusers import DDIMScheduler", source)
        self.assertIn("regiter_attention_editor_diffusers(pipe, AttentionBase())", source)
        self.assertIn("[ref_prompt, prompt]", source)
        self.assertNotIn("from src.diffusers_model_pipeline import CustomDiffusionPipeline", source)
        self.assertNotIn("pytorch_lightning", source)

    def test_runners_use_lightweight_shared_seeding(self):
        common = (ROOT / "benchmark/dreammatcher_vico_common.py").read_text()
        self.assertIn("def seed_everything(seed: int)", common)
        for relative in (
            "benchmark/run_custom_diffusion_vico.py",
            "benchmark/run_dreammatcher_custom_diffusion_vico.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn("seed_everything", source)
            self.assertNotIn("from pytorch_lightning", source)

    def test_dreammatcher_inverts_before_registering_attention_editor(self):
        source = (ROOT / "benchmark/run_dreammatcher_custom_diffusion_vico.py").read_text()
        inversion = source.index("start_code, latents_list = model.invert(")
        prompt_loop = source.index("for prompt_index, template in enumerate")
        registration = source.index("regiter_attention_editor_diffusers(model, editor)")
        self.assertLess(inversion, prompt_loop)
        self.assertLess(prompt_loop, registration)

    def test_launchers_use_shared_evaluator_with_five_images(self):
        for relative in (
            "scripts/run_custom_diffusion_vico_5seeds.sh",
            "scripts/run_dreammatcher_custom_diffusion_vico_5seeds.sh",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn("benchmark/evaluate_vico_arbooth_protocol.py", source)
            self.assertIn("--expected-images-per-prompt 5", source)

    def test_smoke_launchers_use_one_subject_two_prompts_and_five_seeds(self):
        for relative in (
            "scripts/smoke_csd_var_vico_5seeds.sh",
            "scripts/smoke_custom_diffusion_vico_5seeds.sh",
            "scripts/smoke_dreammatcher_custom_diffusion_vico_5seeds.sh",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn('SUBJECT="${SUBJECT:-brown_dog}"', source)
            self.assertIn("MAX_PROMPTS=2", source)
            self.assertIn("0,1,2,3,4", source)


if __name__ == "__main__":
    unittest.main()
