"""CPU integration test with a RANDOM tiny LM; never a task performance result.

Uses real Transformers generation, teacher scoring, optimizer, checkpoint/resume,
and eval CLIs. Creates all artifacts in a new directory and writes a report.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from verifiable_opd.data import build_splits, export, expert_messages, canonical
from verifiable_opd.environment import Task
from verifiable_opd.prompts import TOOL_SCHEMA


TEMPLATE = """{% if tools %}{{ '<|im_start|>system\\nTools: ' + (tools | tojson) + '<|im_end|>\\n' }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}{% if message.get('tool_calls') %}{% for call in message['tool_calls'] %}{{ '<tool_call>' + (call['function'] | tojson) + '</tool_call>' }}{% endfor %}{% else %}{{ message['content'] }}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    splits = build_splits(3, 1, 1, 1)
    export(output / "data", splits, 17)
    corpus = [canonical(expert_messages(Task.from_dict(row["task"]))) for row in splits["train"]]
    corpus += [canonical(TOOL_SCHEMA), "assistant system user tool Tools: <tool_call></tool_call>"]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[UNK]", "<|im_start|>", "<|im_end|>"])
    tok.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
        bos_token="<|im_start|>", eos_token="<|im_end|>", chat_template=TEMPLATE)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=4096, n_embd=32, n_layer=1, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    for name in ("student", "teacher"):
        model = GPT2LMHeadModel(config)
        model.save_pretrained(output / name)
        tokenizer.save_pretrained(output / name)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    commands = []
    def run(module, extra):
        command = [sys.executable, "-m", module, *map(str, extra)]
        commands.append(command)
        proc = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (output / "execution.log").open("a") as f:
            f.write("COMMAND " + json.dumps(command) + "\n" + proc.stdout + "\n")
        if proc.returncode:
            raise RuntimeError(f"failed: {module}; see {output / 'execution.log'}\n{proc.stdout[-3000:]}")
    common = ["--data", output / "data/train.jsonl", "--batch-size", 1, "--max-new-tokens", 2,
              "--max-context", 4096, "--steps", 1, "--lr", .001]
    run("verifiable_opd.reference_train", ["--mode", "sft", "--student", output / "student",
        "--output", output / "sft", *common])
    warm = output / "sft/step_000001/model"
    for mode in ("opd", "uniform", "failure"):
        run("verifiable_opd.reference_train", ["--mode", mode, "--student", warm,
            "--teacher", output / "teacher", "--output", output / mode, "--replay-tokens", 1024, *common])
    run("verifiable_opd.reference_train", ["--mode", "failure", "--student", warm,
        "--teacher", output / "teacher", "--output", output / "failure", "--replay-tokens", 1024,
        *common, "--steps", 2, "--resume", output / "failure/step_000001"])
    run("verifiable_opd.reference_train", ["--mode", "failure", "--student", warm,
        "--teacher", output / "teacher", "--output", output / "failure_uninterrupted", "--replay-tokens", 1024,
        *common, "--steps", 2])
    resumed = GPT2LMHeadModel.from_pretrained(output / "failure/step_000002/model")
    continuous = GPT2LMHeadModel.from_pretrained(output / "failure_uninterrupted/step_000002/model")
    max_diff = max(float((a - b).abs().max().detach()) for a, b in zip(resumed.parameters(), continuous.parameters()))
    if max_diff > 1e-7:
        raise AssertionError(f"CPU resume diverged from uninterrupted run: {max_diff}")
    for mode in ("sft", "opd", "uniform", "failure"):
        step = 2 if mode == "failure" else 1
        run("verifiable_opd.evaluate", ["--model", output / mode / f"step_{step:06d}/model",
            "--data", output / "data/dev.jsonl", "--output", output / f"eval_{mode}",
            "--max-new-tokens", 2])
    report = {"status": "passed", "backend": "transformers_reference", "model": "random tiny GPT2",
        "not_task_performance_evidence": True, "cpu_resume_max_parameter_diff": max_diff,
        "verified": ["sft", "opd", "uniform replay", "failure replay", "resume", "greedy eval"],
        "commands": commands}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    main()
