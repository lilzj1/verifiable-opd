"""Summarize the predeclared source-length diagnostic, without granting training."""
import argparse
import json
from pathlib import Path


def analyze(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "protocol.json").read_text())
    state = json.loads((directory / "status.json").read_text())
    if state["phase"] != "COMPLETE_REVIEW_REQUIRED":
        raise ValueError("Diagnostic is not complete")
    ids = frozen["ids"]
    result = {"phase": "DIAGNOSTIC_REVIEW_REQUIRED", "allow_training": False,
              "n_paired_questions": len(ids), "cells": {}, "comparisons": {},
              "limitations": frozen["differences_from_source"] + [
                  "Small development subset overlaps prior diagnostics; not independent validation",
                  "No claim that longer evaluation validates shorter OPD training",
              ]}
    predictions = {}
    for name, role, gpu, response in frozen["cells"]:
        if state["cells"][name]["exit_code"] != 0:
            raise ValueError(f"{name}: failed or timed out; inspect raw logs, do not score missing rows as zero")
        protocol = json.loads((directory / name / "protocol.json").read_text())
        expected = dict(role=role, physical_gpu=gpu, ids=ids, max_response_tokens=response,
                        dev_sha256=frozen["dev_sha256"])
        expected.update({k: frozen[k] for k in ("temperature", "top_p", "top_k", "seed", "dtype", "max_model_len")})
        if any(protocol.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{name}: protocol mismatch")
        rows = [json.loads(line) for line in (directory / name / "predictions.jsonl").read_text().split("\n") if line.strip()]
        if [row["example_id"] for row in rows] != ids:
            raise ValueError(f"{name}: IDs or ordering mismatch")
        if any("checker_error" in row or row["domain"] != "math" for row in rows):
            raise ValueError(f"{name}: invalid domain or checker failure")
        scores = [float(row["evaluation"]["score"]) for row in rows]
        if any(score not in (0., 1.) for score in scores):
            raise ValueError("Expected binary math scores")
        lengths = [len(row["response_token_ids"]) for row in rows]
        predictions[name] = scores
        summary = json.loads((directory / name / "summary.json").read_text())
        result["cells"][name] = {
            "n": len(rows), "correct": sum(scores), "accuracy": sum(scores)/len(rows),
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
            "generated_tokens": sum(lengths), "wall_seconds": summary["wall_seconds"],
            "correct_responses_longer_than_8192": sum(s == 1 and n > 8192 for s, n in zip(scores, lengths)),
            "paired_scores": scores,
        }
    for label, a, b in (
        ("teacher_minus_student_8k", "math_8k", "student_8k"),
        ("teacher_minus_student_30k", "math_30k", "student_30k"),
        ("student_30k_minus_8k", "student_30k", "student_8k"),
        ("teacher_30k_minus_8k", "math_30k", "math_8k"),
    ):
        pairs = list(zip(predictions[a], predictions[b]))
        result["comparisons"][label] = {
            "delta_percentage_points": 100*sum(x-y for x, y in pairs)/len(pairs),
            "a_only_correct": sum(x > y for x, y in pairs),
            "b_only_correct": sum(x < y for x, y in pairs),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
