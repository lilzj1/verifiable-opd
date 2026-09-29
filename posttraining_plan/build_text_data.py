"""Deterministic local Math/IF splits; never runs a model or judges test outputs."""
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
import unicodedata
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifiable_opd.posttraining.text_encode import encode_prompt, encode_expert, validate_messages

REVISION = "9e897efe3257599d4300e2d5ee865a1cc714af87"
RAW = ROOT / "data/posttraining_raw_v1"
OUT = ROOT / "data/posttraining_text_v3"
SUFFIX = "Please reason step by step, and put your final answer in \\boxed{}."


def task_text(messages):
    text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    if text.rstrip().endswith(SUFFIX):
        text = text.rstrip()[:-len(SUFFIX)]
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(messages, *, source, index, domain, kind, split, checker=None, response=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    validate_messages(messages)
    item = {"example_id": sha(f"{REVISION}:{source}:{index}"), "source_repo": "BytedTsinghua-SIA/Open-MOPD-Data",
        "source_revision": REVISION, "source_path": source, "source_row": index,
        "domain": domain, "kind": kind, "split": split, "messages": messages,
        "dedup_cluster_id": sha(task_text(messages)), "checker": checker or {}}
    if response is not None:
        item["expert_response"] = response
    return item


def write_jsonl(path, rows):
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoTokenizer
    source_manifest = json.loads((RAW / "SOURCE_MANIFEST.json").read_text())
    if source_manifest["phase"] != "RAW_FILES_READY_LICENSE_AND_SPLIT_AUDIT_PENDING":
        raise ValueError("Raw downloads not complete")
    OUT.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/posttraining_assets_20260926_resume_v4/models/shared_student_initialization",
        local_files_only=True, fix_mistral_regex=False)
    test = []
    for source in ("eval/math/aime24.parquet", "eval/math/aime25.parquet",
                   "eval/if/ifeval_aligned.parquet", "eval/if/ifbench_test_aligned.parquet"):
        for i, row in enumerate(pq.read_table(RAW / source).to_pylist()):
            meta = json.loads(row["metadata"])
            domain = "math" if "/math/" in source else "if"
            checker = {"type": row["evaluator"], "answer": row.get("answer"), "metadata": meta,
                       "original_prompt": row["original_prompt"]}
            test.append(record(row["prompt"], source=source, index=i, domain=domain,
                               kind="eval", split="test", checker=checker))
    counts = Counter()
    online = {"math": [], "if": []}
    seen = {r["dedup_cluster_id"] for r in test}
    mapping = {"math_dapo_boxed": "math", "nemotron_if_rl": "if"}
    source = "rl_prompt_mix/train.parquet"
    # Column whitelist prevents reading the code task payloads into prompt construction.
    for batch in pq.ParquetFile(RAW / source).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            counts[f"raw_{row['domain']}"] += 1
            if row["domain"] == "code":
                continue
            if mapping.get(row["data_source"]) != row["domain"]:
                raise ValueError("Unknown or conflicting domain/source mapping")
            extra = row["extra_info"]
            checker = {"type": "math" if row["domain"] == "math" else "nemotron_if",
                "answer": row["reward_model"]["ground_truth"],
                "instruction_id_list": extra.get("instruction_id_list"),
                "kwargs": [json.loads(x) for x in extra.get("instruction_kwargs_json") or []]}
            item = record(row["prompt"], source=source, index=sum(counts[f"raw_{k}"] for k in ("math", "if", "code"))-1,
                          domain=row["domain"], kind="online", split="pending", checker=checker)
            if item["dedup_cluster_id"] in seen:
                counts["online_exact_duplicate_or_test_overlap"] += 1
                continue
            seen.add(item["dedup_cluster_id"])
            online[item["domain"]].append(item)
    # Candidate order depends only on identity and seed, never model results.
    candidates = {domain: sorted(rows, key=lambda r: sha("17:" + r["example_id"]))[:2048]
                  for domain, rows in online.items()}
    texts = [task_text(r["messages"]) for r in test]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                                dtype=np.float32, max_features=250000)
    candidate_rows = candidates["math"] + candidates["if"]
    matrix = vectorizer.fit_transform(texts + [task_text(r["messages"]) for r in candidate_rows])
    protected = list(range(len(test)))
    chosen = {"math": {"dev": [], "train": []}, "if": {"dev": [], "train": []}}
    for j, item in enumerate(candidate_rows, start=len(test)):
        domain = item["domain"]
        split = "dev" if len(chosen[domain]["dev"]) < 64 else "train"
        if len(chosen[domain][split]) >= (64 if split == "dev" else 512):
            continue
        # v2's top-20 semantic review found the same underlying source tasks
        # with altered IF constraints below cosine .90. Keep dev fixed and
        # remove these task families from TRAIN, replenishing by original order.
        if split == "train" and domain == "if" and any(
            marker in task_text(item["messages"]).casefold()
            for marker in ("white 1996: 73", "hypnotizing david", "day 1 script for the ravager")
        ):
            counts["if_train_reviewed_task_family_overlap"] += 1
            continue
        similarity = (matrix[j] @ matrix[protected].T).max()
        if float(similarity) >= 0.90:
            counts["online_near_duplicate_excluded"] += 1
            continue
        try:
            item["prompt_tokens"] = len(encode_prompt(tokenizer, item["messages"]))
        except ValueError:
            counts["online_overlong_prompt"] += 1
            continue
        item["split"] = split
        chosen[domain][split].append(item)
        # Protect against variants among both dev and train, not just across splits.
        protected.append(j)
    for domain in chosen:
        if (len(chosen[domain]["dev"]), len(chosen[domain]["train"])) != (64, 512):
            raise ValueError("Not enough eligible candidates; do not silently shrink the protocol")
    dev = chosen["math"]["dev"] + chosen["if"]["dev"]
    train = []
    for i in range(0, 512, 2):
        train.extend(chosen["math"]["train"][i:i+2] + chosen["if"]["train"][i:i+2])
    protected_rows = test + dev
    protected_hashes = {r["dedup_cluster_id"] for r in protected_rows}
    protective_matrix = vectorizer.transform([task_text(r["messages"]) for r in protected_rows])
    expert = []
    for domain, source in [("math", "sft/openr1_math_93k/train.parquet"),
                           ("if", "sft/instruction_nemotron_aligned/train-00000.parquet")]:
        def eligible_rows():
            index = 0
            for batch in pq.ParquetFile(RAW / source).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    messages = row["messages"]
                    current = index
                    index += 1
                    if len(messages) not in (2, 3) or messages[-1]["role"] != "assistant":
                        counts[f"{domain}_expert_multiturn_excluded"] += 1
                        continue
                    if any(m["role"] not in {"system", "user"} for m in messages[:-1]):
                        continue
                    # Published chat SFT contains provider self-identification.
                    # It is not instruction-following supervision for our model;
                    # omit rather than relabel the original provider as ourselves.
                    response = messages[-1]["content"]
                    if domain == "if" and re.search(
                        r"(?:I(?:'m| am)|My name is).{0,100}(?:NVIDIA|Nemotron)", response, re.I
                    ):
                        counts["if_provider_identity_excluded"] += 1
                        continue
                    yield record(messages[:-1], source=source, index=current, domain=domain,
                        kind="expert", split="train", response=messages[-1]["content"])
        shortlist = heapq.nsmallest(4096, eligible_rows(), key=lambda r: sha("17:" + r["example_id"]))
        selected, expert_seen = [], set()
        for item in shortlist:
            key = item["dedup_cluster_id"]
            if key in protected_hashes or key in expert_seen:
                counts[f"{domain}_expert_exact_overlap"] += 1
                continue
            query = vectorizer.transform([task_text(item["messages"])])
            if float((query @ protective_matrix.T).max()) >= 0.90:
                counts[f"{domain}_expert_near_overlap"] += 1
                continue
            try:
                encoded = encode_expert(tokenizer, item["messages"], item["expert_response"])
            except ValueError as exc:
                counts[f"{domain}_expert_excluded:{exc}"] += 1
                continue
            item.update(prompt_tokens=len(encoded.prompt_ids), response_tokens=len(encoded.response_ids))
            selected.append(item)
            expert_seen.add(key)
            if len(selected) == 128:
                break
        if len(selected) != 128:
            raise ValueError(f"Insufficient complete {domain} demonstrations")
        expert.extend(selected)
    for name, rows in [("dev.jsonl", dev), ("online_train.jsonl", train),
                       ("expert_train.jsonl", expert), ("test_manifest.jsonl", test)]:
        write_jsonl(OUT / name, rows)
    # Labels only in metadata; online raw_prompt is constructed from messages only.
    for name, rows in [("online_train.parquet", train), ("dev.parquet", dev)]:
        pq.write_table(pa.Table.from_pylist([{"data_source": r["domain"], "prompt": r["messages"],
            "ability": r["domain"], "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"example_id": r["example_id"], "split": r["split"],
                           "checker_json": json.dumps(r["checker"]), "domain": r["domain"]}}
            for r in rows]), OUT / name)
    manifest = {"phase": "SPLITS_BUILT_NEAR_DUPLICATE_AUDIT_REQUIRED", "seed": 17,
        "source_revision": REVISION, "domain_mapping": mapping, "counts": dict(counts),
        "sizes": {"train": len(train), "dev": len(dev), "expert": len(expert), "test": len(test)},
        "near_dedup": "NFC/whitespace, exact hash + case-sensitive char(3,5) TF-IDF cosine >=0.90; heuristic not proof of no contamination",
        "expert_length": {d: [r["response_tokens"] for r in expert if r["domain"] == d] for d in chosen},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}}
    (OUT / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"phase": manifest["phase"], "sizes": manifest["sizes"], "counts": manifest["counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
