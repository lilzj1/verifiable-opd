import json
from pathlib import Path
import tempfile
import unittest

from analyze_source_length_diagnostic import analyze


class SourceDiagnosticTests(unittest.TestCase):
    def fixture(self, root):
        cells = [("student_8k", "shared_student_initialization", "1", 8192),
                 ("math_8k", "math_teacher", "2", 8192),
                 ("student_30k", "shared_student_initialization", "3", 30000),
                 ("math_30k", "math_teacher", "4", 30000)]
        common = dict(dev_sha256="frozen", temperature=.6, top_p=.95, top_k=-1,
                      seed=17, dtype="float16", max_model_len=32768, ids=["a", "b"])
        (root / "protocol.json").write_text(json.dumps(dict(common, cells=cells, differences_from_source=[])))
        state = dict(phase="COMPLETE_REVIEW_REQUIRED", cells={name: {"exit_code": 0} for name, *_ in cells})
        (root / "status.json").write_text(json.dumps(state))
        for name, role, gpu, response in cells:
            folder = root / name
            folder.mkdir()
            (folder / "protocol.json").write_text(json.dumps(dict(common, role=role, physical_gpu=gpu, max_response_tokens=response)))
            rows = [dict(example_id=key, domain="math", evaluation={"score": int(name == "math_30k")},
                         response_token_ids=[1, 2], finish_reason="stop") for key in common["ids"]]
            (folder / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (folder / "summary.json").write_text(json.dumps({"wall_seconds": 1}))

    def test_paired_summary_does_not_authorize_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = analyze(root)
            self.assertFalse(result["allow_training"])
            self.assertEqual(result["comparisons"]["teacher_minus_student_30k"]["a_only_correct"], 2)

    def test_rejects_failed_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / "status.json"
            state = json.loads(path.read_text())
            state["cells"]["math_30k"]["exit_code"] = 124
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "failed or timed out"):
                analyze(root)

    def test_rejects_changed_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / "math_30k/protocol.json"
            protocol = json.loads(path.read_text())
            protocol["top_k"] = 20
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
