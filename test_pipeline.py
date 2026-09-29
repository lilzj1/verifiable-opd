import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from verifiable_opd.data import build_splits, export, read_records, expert_messages, make_record
from verifiable_opd.environment import Task, ArithmeticEnv
from verifiable_opd.hf_backend import rollout
from verifiable_opd.model_integrity import validate_local_model
from verifiable_opd.teacher_sft import collate, labeled_example
from verifiable_opd.replay import select_replay
from verifiable_opd.trajectory import parse_action, run_episode
from verifiable_opd.tokenization import causal_target_slice


class PipelineTests(unittest.TestCase):
    def test_model_integrity_rejects_index_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Config/tokenizer are checked only after shard/index agreement, so
            # a corrupt weight map must fail first and without model loading.
            save_file({"weight": torch.ones(1)}, root / "model-00001-of-00001.safetensors")
            (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"different": "model-00001-of-00001.safetensors"}
            }))
            with self.assertRaisesRegex(ValueError, "index"):
                validate_local_model(root)

    def test_teacher_sft_masks_prompt_and_padding(self):
        class Decision:
            prefix = [1, 2]
            response = [3, 4]

        row = labeled_example(Decision())
        self.assertEqual(row, {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]})
        batch = collate([row, {"input_ids": [5], "labels": [5]}], 0)
        self.assertEqual(batch["input_ids"].tolist()[1], [5, 0, 0, 0])
        self.assertEqual(batch["labels"].tolist()[1], [5, -100, -100, -100])
        self.assertEqual(batch["attention_mask"].tolist()[1], [1, 0, 0, 0])

    def test_reference_rollout_stops_after_one_tool_call(self):
        class Tokenizer:
            eos_token_id = 9
            pad_token_id = 9

            def apply_chat_template(self, *args, **kwargs):
                return [1, 2]

            def decode(self, *args, **kwargs):
                return '<tool_call>{"name":"submit_value","arguments":{"value":5}}</tool_call>'

        class Model:
            def __init__(self):
                self.kwargs = None

            def eval(self):
                return self

            def parameters(self):
                yield torch.zeros(1)

            def generate(self, input_ids, attention_mask, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[*input_ids[0].tolist(), 3]])

        model, tokenizer = Model(), Tokenizer()
        row = make_record(Task(3, (("add", 2),)), "dev")
        _, trace = rollout(model, tokenizer, row, sample=False)
        self.assertTrue(trace["success"])
        self.assertEqual(model.kwargs["stop_strings"], ["</tool_call>"])
        self.assertIs(model.kwargs["tokenizer"], tokenizer)

    def test_disjoint_reproducible_length_splits(self):
        splits = build_splits(9, 6, 6, 4)
        self.assertEqual(splits, build_splits(9, 6, 6, 4))
        all_ids = [r["task_id"] for rows in splits.values() for r in rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual({r["length"] for r in splits["test_ood"]}, {12, 16})
        self.assertEqual({r["length"] for r in splits["train"]}, {3, 5, 8})

    def test_export_integrity_and_test_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "data"
            export(out, build_splits(3, 1, 1, 1), 17)
            self.assertEqual(len(read_records(out / "train.jsonl", {"train"})), 3)
            with self.assertRaises(ValueError):
                read_records(out / "test_id.jsonl", {"train"})
            with self.assertRaises(FileExistsError):
                export(out, build_splits(3, 1, 1, 1), 17)
            self.assertFalse((out / "sft_test_id.jsonl").exists())

    def test_experts_are_executable(self):
        for row in build_splits(12, 1, 1, 1)["train"]:
            task = Task.from_dict(row["task"])
            env = ArithmeticEnv(task)
            for message in expert_messages(task):
                if message["role"] == "assistant":
                    args = message["tool_calls"][0]["function"]["arguments"]
                    env.step(json.dumps(args))
            self.assertTrue(env.success)

    def test_parquet_round_trip(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data"
            splits = build_splits(3, 1, 1, 1)
            export(output, splits, 17, parquet=True)
            rows = pq.read_table(output / "train.parquet").to_pylist()
            for record, expected in zip(rows, splits["train"]):
                payload = record["extra_info"]["tools_kwargs"]["submit_value"]["create_kwargs"]
                self.assertEqual(json.loads(payload["task_json"]), expected["task"])
            sft = pq.read_table(output / "sft_train.parquet").to_pylist()
            self.assertEqual(len(sft), 3)
            self.assertTrue(any(m.get("tool_calls") for m in sft[0]["messages"]))

    def test_failure_budget_and_uniform_fallback(self):
        costs = {"a": 100, "b": 200, "c": 100}
        pick = select_replay(costs, {"b"}, 200, 7, "failure")
        self.assertEqual(pick["task_ids"], ["b"])
        self.assertEqual(pick["tokens"], 200)
        pick = select_replay(costs, set(), 200, 7, "failure")
        self.assertEqual(pick["failure_selected"], 0)
        self.assertGreater(pick["uniform_fallback"], 0)
        with self.assertRaises(ValueError):
            select_replay(costs, {"test-id"}, 200, 7)

    def test_multiple_calls_rejected(self):
        call = '<tool_call>{"name":"submit_value","arguments":{"value":5}}</tool_call>'
        self.assertEqual(parse_action(call).parameters, {"value": 5})
        self.assertIsNotNone(parse_action(call + call).error)
        self.assertIsNotNone(parse_action('<tool_call>[]</tool_call>').error)

    def test_episode_succeeds_only_for_full_chain(self):
        row = make_record(Task(3, (("add", 2), ("mul", 4))), "dev")
        values = iter([5, 20])
        def expert(_):
            return '<tool_call>' + json.dumps({"name": "submit_value", "arguments": {"value": next(values)}}) + '</tool_call>'
        self.assertTrue(run_episode(row, expert)["success"])
        bad = run_episode(row, lambda _: "malformed")
        self.assertFalse(bad["success"])
        self.assertEqual(bad["calls"], 4)

    def test_shift(self):
        self.assertEqual(list(range(10))[causal_target_slice(4, 3)], [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
