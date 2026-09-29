"""Rule-based local dev checkers; public final-test adapters are separate."""
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def final_text(response):
    if "<think>" in response and "</think>" not in response:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, count=1, flags=re.DOTALL).strip()


def score(row, response):
    final = final_text(response)
    checker = row["checker"]
    if checker["type"] == "math":
        from math_verify import parse, verify
        gold = parse(str(checker["answer"]))
        if not gold:
            raise ValueError("Unparseable gold answer, not a model failure")
        parsed = parse(final) if final else []
        return {"score": float(bool(parsed) and verify(gold, parsed)),
                "parse_valid": bool(parsed), "final_text": final}
    if checker["type"] != "nemotron_if":
        raise ValueError("This checker only supports local dev; public test needs frozen official adapters")
    path = ROOT / "upstream/open-mopd-reference/training/third_party/verifiable-instructions"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import nltk
    from langdetect import DetectorFactory
    from verifiable_instructions.instructions_registry import INSTRUCTION_DICT
    nltk.data.path.insert(0, str(ROOT / "data/nltk"))
    DetectorFactory.seed = 17
    random.seed(17)
    flags = []
    ids, kwargs = checker["instruction_id_list"], checker["kwargs"]
    if not ids or len(ids) != len(kwargs):
        raise ValueError("Invalid IF checker metadata")
    for key, arguments in zip(ids, kwargs, strict=True):
        instruction = INSTRUCTION_DICT[key](key)  # Unknown constraint is infrastructure failure.
        instruction.build_description(**(arguments or {}))
        if "prompt" in (instruction.get_instruction_args() or {}):
            instruction.build_description(prompt=row["messages"][-1]["content"])
        flags.append(bool(final.strip()) and bool(instruction.check_following(final)))
    return {"score": float(all(flags)), "instruction_flags": flags,
            "instruction_accuracy": sum(flags) / len(flags), "final_text": final,
            "exact_empty_think_format": response.lstrip().startswith("<think>\n</think>")}
