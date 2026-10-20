"""CPU-only contract checks for the CSD-VAR ViCo adapter."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("csd_vico", ROOT / "benchmark/run_csd_var_vico.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def test_prepare_subject_and_normalize_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ref.jpg").write_bytes(b"reference")
            item = {"subject": "brown_dog", "class_name": "dog", "single_reference": "ref.jpg",
                    "prompt_templates": ["a {} in snow", "a {} on grass"]}
            data, prompts = adapter.prepare_subject(root, item, root / "work", 1)
            self.assertEqual((data / "imgs/reference.jpg").read_bytes(), b"reference")
            with (data / "captions.csv").open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows[1]), 5)
            self.assertEqual(rows[1][1], "reference.jpg")
            self.assertEqual(len((data / "variation_content.txt").read_text().splitlines()), 10)
            self.assertEqual(json.loads(prompts.read_text())["content"]["recontextualization"], ["a {} dog in snow"])
            source = root / "raw/example/output/ours/prompt_0_content_recontextualization"
            source.mkdir(parents=True)
            for i in range(2):
                (source / f"{i}.png").write_bytes(str(i).encode())
            target = root / "output"
            adapter.normalize_outputs(root / "raw", target, 1, 2)
            self.assertEqual((target / "prompt_00/seed_001.png").read_bytes(), b"1")
            with self.assertRaises(RuntimeError):
                adapter.normalize_outputs(root / "raw", target, 1, 2)

    def test_subject_selection(self):
        manifest = {"subjects": [{"subject": "a"}, {"subject": "b"}]}
        self.assertEqual(adapter.selected_subjects(manifest, "b"), [{"subject": "b"}])
        with self.assertRaises(ValueError):
            adapter.selected_subjects(manifest, "missing")

    def test_checkpoint_names(self):
        names = {p.name for p in adapter.checkpoint_files(Path("ckpt"), 150)}
        self.assertIn("virtual_token_embed-steps-150.pt", names)
        self.assertIn("learned_embeds_2-steps-150_style.safetensors", names)
        self.assertEqual(len(names), 5)

    def test_one_training_and_eight_samples_per_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.jpg"
            ref.write_bytes(b"reference")
            item = {"subject": "dog", "class_name": "dog", "single_reference": str(ref),
                    "prompt_templates": ["a {} in snow", "a {} on grass"]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"subjects": [item]}))
            work, output = root / "work", root / "outputs"
            calls = []

            def fake_run(command, cwd):
                calls.append(command)
                data = work / "dog/data/dog+photo"
                if "extract_svd.py" in command:
                    for name in ("proj_top_10.pt", "proj_2_top_10.pt"):
                        (data / name).touch()
                elif "train.py" in command:
                    checkpoint = work / "dog/checkpoint"
                    checkpoint.mkdir(parents=True)
                    for path in adapter.checkpoint_files(checkpoint, 150):
                        path.touch()
                elif "inference.py" in command:
                    self.assertEqual(command[command.index("--images_per_prompt") + 1], "8")
                    self.assertEqual(command[command.index("--seed") + 1], "42")
                    for prompt in range(2):
                        dst = work / f"dog/raw/run/output/ours/prompt_{prompt}_content_recontextualization"
                        dst.mkdir(parents=True)
                        for i in range(8):
                            (dst / f"{i}.png").write_bytes(b"png")

            argv = ["run_csd_var_vico.py", "--manifest", str(manifest), "--output", str(output),
                    "--work-dir", str(work)]
            with mock.patch.object(adapter.sys, "argv", argv), mock.patch.object(adapter, "run", side_effect=fake_run):
                adapter.main()
            self.assertEqual(sum("train.py" in call for call in calls), 1)
            self.assertEqual(sum("inference.py" in call for call in calls), 1)
            for prompt in range(2):
                self.assertEqual(len(list((output / f"dog/prompt_{prompt:02d}").glob("seed_*.png"))), 8)
            timing = json.loads((output / "dog/timing.json").read_text())
            self.assertEqual(timing["seed"], 42)
            self.assertIsNone(timing["inference_seconds_per_image"])


if __name__ == "__main__":
    unittest.main()
