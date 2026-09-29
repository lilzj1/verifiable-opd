"""Synthetic CPU contract cases. No real task content or external dependencies."""
import copy
import json
from pathlib import Path
import unittest

from check_contracts import check_draft, check_records, evaluation_response_budget, public_messages


class Contracts(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(Path(__file__).with_name("protocol.draft.json").read_text())
        self.row = {
            "example_id": "synthetic:a", "dedup_cluster_id": "cluster:a",
            "source_revision": "synthetic-v1", "domain": "math", "split": "train",
            "kind": "online", "messages": [{"role": "user", "content": "A synthetic prompt"}],
            "ground_truth": "SECRET_LABEL",
        }

    def test_draft_valid(self):
        check_draft(self.plan)

    def test_draft_cannot_authorize_training(self):
        self.plan["allow_training"] = True
        with self.assertRaises(ValueError):
            check_draft(self.plan)

    def test_context_mismatch(self):
        self.plan["candidate"]["max_response_tokens"] = 8192
        with self.assertRaises(ValueError):
            check_draft(self.plan)

    def test_resource_pool_mismatch(self):
        self.plan["candidate"]["teacher_tp"] = 2
        with self.assertRaises(ValueError):
            check_draft(self.plan)

    def test_s2_has_no_ce(self):
        self.plan["groups"]["S2"]["replay_alpha"] = 0.1
        with self.assertRaises(ValueError):
            check_draft(self.plan)

    def test_eval_budget(self):
        self.assertEqual(evaluation_response_budget(self.plan),
                         {"per_checkpoint": 1081, "all_student_checkpoints": 4324})

    def test_public_input_whitelist(self):
        check_records([self.row])
        self.assertNotIn("SECRET_LABEL", json.dumps(public_messages(self.row)))

    def test_unknown_domain(self):
        self.row["domain"] = "code"
        with self.assertRaises(ValueError):
            check_records([self.row])

    def test_replay_dev_leak(self):
        self.row.update(kind="expert", split="dev", expert_response="answer")
        with self.assertRaises(ValueError):
            check_records([self.row])

    def test_cluster_leak(self):
        other = copy.deepcopy(self.row)
        other.update(example_id="synthetic:b", split="dev", kind="eval")
        with self.assertRaises(ValueError):
            check_records([self.row, other])

    def test_train_online_and_expert_overlap_allowed(self):
        other = copy.deepcopy(self.row)
        other.update(example_id="synthetic:b", kind="expert", expert_response="answer")
        check_records([self.row, other])

    def test_no_assistant_answer_in_input(self):
        self.row["messages"].append({"role": "assistant", "content": "answer"})
        with self.assertRaises(ValueError):
            check_records([self.row])

    def test_no_extra_label_field_in_message(self):
        self.row["messages"][0]["answer"] = "label"
        with self.assertRaises(ValueError):
            check_records([self.row])

    def test_duplicate_id(self):
        with self.assertRaises(ValueError):
            check_records([self.row, copy.deepcopy(self.row)])

    def test_empty_expert_response(self):
        self.row.update(kind="expert", expert_response="")
        with self.assertRaises(ValueError):
            check_records([self.row])


if __name__ == "__main__":
    unittest.main()
