"""CPU checks for the real verl expert-loop, DataProto, and mixed-loss contracts."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.config.distillation import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding

from verifiable_opd.data import canonical, initial_messages, make_record
from verifiable_opd.environment import Task
from verifiable_opd.expert_agent_loop import render_expert_trajectory
from verifiable_opd.verl_mixed_loss import mixed_opd_replay_loss
from verifiable_opd.verl_replay_batch import (
    ReplayPool,
    concat_for_joint_update,
    failed_task_ids,
    make_expert_input,
    restore_rollout_identity,
)


MODEL = (
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


def rl_sample(task: Task, index: int):
    record = make_record(task, "train")
    return {
        "data_source": "verifiable_arithmetic",
        "agent_name": "arithmetic_tool_agent",
        "raw_prompt": initial_messages(task),
        "reward_model": {"style": "rule", "ground_truth": "environment_success"},
        "extra_info": {
            "task_id": record["task_id"],
            "split": "train",
            "tools_kwargs": {
                "submit_value": {
                    "create_kwargs": {
                        "task_json": canonical(task.to_dict()),
                        "mode": "sparse",
                        "max_calls": 2 * len(task.operations),
                    }
                }
            },
        },
        "tools_kwargs": {
            "submit_value": {
                "create_kwargs": {
                    "task_json": canonical(task.to_dict()),
                    "mode": "sparse",
                    "max_calls": 2 * len(task.operations),
                }
            }
        },
        "index": index,
        "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
    }


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples
        self.dataframe = [
            {"extra_info": copy.deepcopy(sample["extra_info"])} for sample in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return copy.deepcopy(self.samples[index])


class VerlReplayTests(unittest.TestCase):
    def test_text_only_trainer_does_not_require_multimodal_key(self):
        trainer = Path("upstream/verl-v0.8.0/verl/trainer/ppo/ray_trainer.py").read_text()
        self.assertIn('batch.non_tensor_batch.get("multi_modal_inputs", [])', trainer)
        self.assertIn('.get("mfu", 0.0)', trainer)

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def test_exact_expert_render_and_mask(self):
        task = Task(3, (("add", 2), ("mul", 4)))
        prompt, response, mask, turns = render_expert_trajectory(self.tokenizer, task)
        self.assertEqual(turns, 2)
        self.assertGreater(len(prompt), 0)
        self.assertEqual(len(response), len(mask))
        action_text = self.tokenizer.decode([token for token, keep in zip(response, mask) if keep])
        context_text = self.tokenizer.decode([token for token, keep in zip(response, mask) if not keep])
        self.assertEqual(action_text.count("<tool_call>"), 2)
        self.assertIn('"value": 5', action_text)
        self.assertIn('"value": 20', action_text)
        self.assertNotIn("tool_response", action_text)
        self.assertIn("tool_response", context_text)

    def test_pool_failure_selection_and_expert_dataproto(self):
        samples = [
            rl_sample(Task(3 + i, (("add", 2), ("mul", 4))), i) for i in range(3)
        ]
        dataset = FakeDataset(samples)
        pool = ReplayPool.from_dataset(dataset, self.tokenizer)
        failed_id = samples[1]["extra_info"]["task_id"]
        selection = pool.select(
            {failed_id}, budget=sum(pool.costs.values()), max_trajectories=2,
            seed=17, strategy="failure"
        )
        self.assertEqual(selection["task_ids"][0], failed_id)
        expert = make_expert_input(
            dataset, collate_fn, selection, global_steps=9, temperature=0.75
        )
        self.assertEqual(len(expert), 2)
        self.assertEqual(expert.meta_info["global_steps"], 9)
        self.assertEqual(expert.meta_info["temperature"], 0.75)
        self.assertEqual(expert.to_tensordict()["temperature"], 0.75)
        self.assertTrue(all(name == "arithmetic_expert_replay_agent"
                            for name in expert.non_tensor_batch["agent_name"]))
        self.assertTrue(all(str(uid).startswith("expert-replay-9-")
                            for uid in expert.non_tensor_batch["uid"]))

        online = DataProto.from_single_dict(collate_fn(samples[:2]))
        online.non_tensor_batch["environment_success"] = np.array([True, False], dtype=object)
        self.assertEqual(failed_task_ids(online), {failed_id})

    def test_restore_rollout_identity_from_generation_input(self):
        source = DataProto.from_dict(
            tensors={"dummy": torch.zeros(2, 1)},
            non_tensors={
                "extra_info": np.array(
                    [{"task_id": "a"}, {"task_id": "b"}], dtype=object
                )
            },
        )
        output = DataProto.from_dict(tensors={"dummy": torch.ones(2, 1)})
        restored = restore_rollout_identity(output, source)
        self.assertEqual(
            [item["task_id"] for item in restored.non_tensor_batch["extra_info"]],
            ["a", "b"],
        )
        source.non_tensor_batch["extra_info"][0]["task_id"] = "changed"
        self.assertEqual(restored.non_tensor_batch["extra_info"][0]["task_id"], "a")

    def test_joint_concat_zero_fills_advantage_without_changing_online_values(self):
        def proto(rows):
            return DataProto.from_dict(
                tensors={
                    "responses": torch.ones(rows, 3, dtype=torch.long),
                    "response_mask": torch.ones(rows, 3, dtype=torch.long),
                },
                non_tensors={"task": np.array([f"t{i}" for i in range(rows)], dtype=object)},
            )

        online, expert = proto(2), proto(1)
        online.batch["advantages"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        combined = concat_for_joint_update(online, expert)
        self.assertEqual(len(combined), 3)
        torch.testing.assert_close(combined.batch["advantages"][:2],
                                   torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        torch.testing.assert_close(combined.batch["advantages"][2], torch.zeros(3))
        self.assertEqual(combined.batch["expert_replay"].tolist(), [False, False, True])

    def test_mixed_loss_backpropagates_online_and_expert_terms(self):
        padded = TensorDict(
            {
                "prompts": torch.tensor([[1, 2], [1, 2]]),
                "responses": torch.tensor([[3, 4, 0], [5, 0, 0]]),
                "input_ids": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
                "position_ids": torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 0, 0]]),
                "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                "old_log_probs": torch.tensor([[-2.0, -3.0, 0.0], [-4.0, 0.0, 0.0]]),
                "teacher_logprobs": torch.tensor(
                    [[-1.0, -1.0, -1.0, -2.5, 0.0], [-1.0, -1.0, -3.0, 0.0, 0.0]]
                ),
                "teacher_ids": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 0, 0]]),
                "expert_replay": torch.tensor([False, True]),
            },
            batch_size=[2],
        )
        data = left_right_2_no_padding(padded)
        tu.assign_non_tensor(
            data, dp_size=1, opd_online_action_tokens=2, opd_expert_action_tokens=1
        )
        values = torch.tensor([-9., -9., -2., -3., -9., -9., -4.], requires_grad=True)
        log_probs = torch.nested.nested_tensor_from_jagged(
            values, offsets=data["input_ids"].offsets()
        )
        loss_config = DistillationLossConfig(
            loss_mode="k1", use_policy_gradient=True, use_task_rewards=False,
            expert_replay_strategy="uniform", expert_replay_alpha=0.2,
            expert_replay_token_budget=10, expert_replay_max_trajectories=1,
        )
        loss, metrics = mixed_opd_replay_loss(
            OmegaConf.create({"loss_agg_mode": "token-mean", "loss_scale_factor": None}),
            SimpleNamespace(distillation_loss=loss_config),
            model_output={"log_probs": log_probs}, data=data,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("expert_replay/ce", metrics)
        loss.backward()
        self.assertNotEqual(values.grad[1].item(), 0.0)  # online OPD token
        self.assertAlmostEqual(values.grad[5].item(), -0.2, places=6)  # expert CE token

    def test_mixed_loss_matches_ddp_when_ranks_hold_different_row_types(self):
        def padded():
            return TensorDict(
                {
                    "prompts": torch.tensor([[1, 2], [1, 2]]),
                    "responses": torch.tensor([[3, 4, 0], [5, 0, 0]]),
                    "input_ids": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 0, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
                    "position_ids": torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 0, 0]]),
                    "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                    "old_log_probs": torch.tensor([[-2.0, -3.0, 0.0], [-4.0, 0.0, 0.0]]),
                    "teacher_logprobs": torch.tensor(
                        [[-1.0, -1.0, -1.0, -2.5, 0.0], [-1.0, -1.0, -3.0, 0.0, 0.0]]
                    ),
                    "teacher_ids": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 0, 0]]),
                    "expert_replay": torch.tensor([False, True]),
                },
                batch_size=[2],
            )

        loss_config = DistillationLossConfig(
            loss_mode="k1",
            use_policy_gradient=True,
            use_task_rewards=False,
            expert_replay_strategy="uniform",
            expert_replay_alpha=0.2,
            expert_replay_token_budget=10,
            expert_replay_max_trajectories=1,
        )
        actor_config = OmegaConf.create(
            {"loss_agg_mode": "token-mean", "loss_scale_factor": None}
        )
        distillation = SimpleNamespace(distillation_loss=loss_config)

        def calculate(rows, values, dp_size):
            data = left_right_2_no_padding(rows)
            tu.assign_non_tensor(
                data,
                dp_size=dp_size,
                opd_online_action_tokens=2,
                opd_expert_action_tokens=1,
            )
            flat = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            log_probs = torch.nested.nested_tensor_from_jagged(
                flat, offsets=data["input_ids"].offsets()
            )
            loss, _ = mixed_opd_replay_loss(
                actor_config,
                distillation,
                model_output={"log_probs": log_probs},
                data=data,
            )
            return loss, flat

        full_loss, full_values = calculate(
            padded(), [-9, -9, -2, -3, -9, -9, -4], dp_size=1
        )
        online_loss, online_values = calculate(
            padded()[:1], [-9, -9, -2, -3], dp_size=2
        )
        expert_loss, expert_values = calculate(
            padded()[1:], [-9, -9, -4], dp_size=2
        )
        simulated_ddp_loss = (online_loss + expert_loss) / 2
        torch.testing.assert_close(full_loss, simulated_ddp_loss)

        full_loss.backward()
        simulated_ddp_loss.backward()
        torch.testing.assert_close(
            full_values.grad,
            torch.cat([online_values.grad, expert_values.grad]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
