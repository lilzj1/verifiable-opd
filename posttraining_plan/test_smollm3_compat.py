import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from smollm3_compat import ROLES, make_overlay, normalize, verify_effective

ROOT = Path(__file__).resolve().parent.parent


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((ROOT / "artifacts/config_compat_v1/reference4576/report.json").read_text())

    def body(self, role):
        return (ROOT / f"artifacts/two_gpu_preflight_v1/{role}.config.json").read_bytes()

    def test_all_normalize_without_unrelated_edits(self):
        for role in ROLES:
            body = self.body(role)
            source = json.loads(body)
            new, manifest = normalize(body, role, self.reference)
            self.assertEqual(new["rope_theta"], 2_000_000)
            self.assertEqual(new["rope_parameters"]["rope_theta"], 2_000_000)
            for key in set(source) - {"rope_theta", "rope_scaling", "rope_parameters", "tie_word_embeddings"}:
                self.assertEqual(new[key], source[key])
            self.assertEqual(manifest["source_sha256"], ROLES[role])

    def test_changed_source_rejected(self):
        with self.assertRaises(ValueError):
            normalize(self.body("math_teacher") + b" ", "math_teacher", self.reference)

    def test_wrong_reference_rejected(self):
        ref = copy.deepcopy(self.reference)
        ref["transformers"] = "5.3.0"
        with self.assertRaises(ValueError):
            normalize(self.body("math_teacher"), "math_teacher", ref)

    def test_missing_reference_rejected(self):
        ref = copy.deepcopy(self.reference)
        ref["models"] = []
        with self.assertRaises(ValueError):
            normalize(self.body("math_teacher"), "math_teacher", ref)

    def test_runtime_guard(self):
        new, _ = normalize(self.body("math_teacher"), "math_teacher", self.reference)
        verify_effective(SimpleNamespace(**new))
        new["rope_parameters"]["rope_theta"] = 5_000_000
        with self.assertRaises(ValueError):
            verify_effective(SimpleNamespace(**new))

    def test_overlay_partial_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            (source / "config.json").write_bytes(self.body("math_teacher"))
            with self.assertRaises(ValueError):
                make_overlay(source, Path(folder) / "out", "math_teacher", self.reference)
            self.assertFalse((Path(folder) / "out").exists())

    def test_overlay_preserves_source_and_rejects_overwrite(self):
        # Synthetic placeholder assets only test filesystem contracts, not loading.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            body = self.body("math_teacher")
            (source / "config.json").write_bytes(body)
            for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                (source / name).write_text("fixture")
            out = Path(folder) / "out"
            make_overlay(source, out, "math_teacher", self.reference)
            self.assertEqual((source / "config.json").read_bytes(), body)
            self.assertTrue((out / "model.safetensors").is_symlink())
            with self.assertRaises(FileExistsError):
                make_overlay(source, out, "math_teacher", self.reference)


if __name__ == "__main__":
    unittest.main()
