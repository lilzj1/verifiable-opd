"""Freeze the agreed 100-update comparison AFTER engineering checks, before test use."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.compile_text_config import build
from posttraining_plan.run_text_smoke import profile_evidence, audit_updates
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, replay_contract


def budget_sft(lengths, target, seed=17):
    """Exact precomputed upstream DistributedSampler(seed, epoch), complete batches.

    Stop at the first whole batch reaching the *profile-estimated* token budget;
    never truncate expert targets, claim equal FLOPs or use final test scores.
    """
    import torch
    from torch.utils.data import DistributedSampler
    if not lengths or len(lengths) % 4 or min(lengths) <= 0 or target <= 0:
        raise ValueError('Positive token counts and complete four-example batches required')
    sampler = DistributedSampler(lengths, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=True)
    total, updates = 0, 0
    for epoch in range(100):
        sampler.set_epoch(epoch)
        order = list(sampler)
        for start in range(0, len(order), 4):
            total += sum(lengths[i] for i in order[start:start+4])
            updates += 1
            if total >= target:
                return dict(updates=updates, target_tokens=target, scheduled_target_tokens=total,
                            batch_size=4, sampler_seed=seed, complete_expert_rows=updates*4,
                            epochs=epoch+1, alignment='Estimated S2 target tokens from fixed three-step profile; not equal FLOPs')
    raise ValueError('SFT budget outside bounded 100-epoch limit')


def engineering_evidence():
    profile_evidence(ROOT/'artifacts')
    folder = ROOT/'artifacts/text_profile_controller_v2'
    if json.loads((folder/'status.json').read_text())['phase'] != 'PROFILES_COMPLETE_S1_AND_FORMAL_BUDGET_PENDING':
        raise ValueError('Both continuous profiles must finish')
    evidence = [folder/'cost_report.json']
    cost = json.loads(evidence[0].read_text())
    for group in ('S2', 'S3'):
        run = ROOT/f'artifacts/text_{group.lower()}_source16k_profile3_v1'
        result = run/'profile_result.json'
        if json.loads(result.read_text())['phase'] != 'THREE_UPDATES_VERIFIED_NO_WEIGHTS_SAVED':
            raise ValueError('Unverified profile')
        audit_updates(run/'optimizer_audit', 3)
        if set(cost[group]['steps']) != {'1','2','3'}:
            raise ValueError('Missing measured profile steps')
        evidence.extend([result, run/'preflight.json'])
        evidence.extend((run/'optimizer_audit').glob('rank_*.jsonl'))
    controller = ROOT/'artifacts/text_s1_smoke_controller_v1'
    if json.loads((controller/'status.json').read_text())['phase'] != 'S1_SMOKE_AND_RESTORE_COMPLETE_FORMAL_NOT_STARTED':
        raise ValueError('S1 distributed smoke and restore required')
    proof = ROOT/'artifacts/text_s1_sft_smoke_v1_restore/restore_result.json'
    restored = json.loads(proof.read_text())
    if restored['phase'] != 'DISTRIBUTED_RESTORE_VERIFIED':
        raise ValueError('S1 actual restore missing')
    evidence.append(proof)
    for name, sha in restored['report_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe audit filename')
        path = proof.parent/'restore_audit'/name
        if file_sha(path) != sha:
            raise ValueError('S1 restore evidence changed')
        evidence.append(path)
    return cost, {str(path.relative_to(ROOT)): file_sha(path) for path in evidence}


def checker_hashes():
    files = list((ROOT/'verifiable_opd/posttraining').glob('*.py'))
    files += [ROOT/'posttraining_plan/eval_text.py']
    for name in ('upstream/open-mopd-reference/training/third_party/verifiable-instructions/verifiable_instructions',
                 'upstream/open-mopd-reference/evals/verifier/score_functions/instruction_following/google_ifeval',
                 'upstream/ifbench-reference/ifbench'):
        paths = list((ROOT/name).rglob('*.py'))
        if not paths:
            raise ValueError(f'Checker source missing: {name}')
        files += paths
    return {str(p.relative_to(ROOT)): file_sha(p) for p in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cost, evidence = engineering_evidence()
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verifiable_opd.posttraining.sft_dataset import TextExpertSFTDataset
    from posttraining_plan.compile_sft_config import build as build_sft
    from posttraining_plan.run_sft_smoke import sft_contract
    configs = {g: build(g, f'text_{g.lower()}_formal_s100_seed17_v1', updates=100, response_length=16384) for g in ('S2','S3')}
    for config in configs.values():
        config.trainer.save_freq = 10
        # CLI false flags were omitted upstream; real verified vLLM uses True.
        # Make that effective behavior explicit rather than claim it was False.
        config.actor_rollout_ref.rollout.enable_prefix_caching = True
        config.actor_rollout_ref.rollout.enable_chunked_prefill = True
        for key in ('math_teacher','if_teacher'):
            config.distillation.teacher_models[key].inference.enable_prefix_caching = True
            config.distillation.teacher_models[key].inference.enable_chunked_prefill = True
    sft = build_sft('text_s1_formal_tokens_seed17_v1')
    tokenizer = AutoTokenizer.from_pretrained(sft.model.path, local_files_only=True, fix_mistral_regex=False)
    dataset = TextExpertSFTDataset(sft.data.train_files, tokenizer, sft.data)
    target = math.ceil(sum(row['response_tokens'] for row in cost['S2']['by_domain'].values())/3*100)
    budget = budget_sft([len(e.response_ids) for e in dataset.encoded], target)
    sft.trainer.total_training_steps = budget['updates']
    sft.trainer.total_epochs = budget['epochs']
    # S1 is last, cheap teacher-forcing; retain one final full checkpoint.
    sft.trainer.save_freq = budget['updates']
    configs['S1'] = sft
    paths = set((ROOT/'posttraining_plan').glob('*.py')) | set((ROOT/'verifiable_opd/posttraining').glob('*.py'))
    paths.add(ROOT/'verifiable_opd/verl_mixed_loss.py')
    paths.add(ROOT/'upstream/verl-v0.8.0/verl/trainer/sft_trainer_ray.py')
    for group in ('S2','S3'):
        preflight = json.loads((ROOT/f'artifacts/text_{group.lower()}_source16k_profile3_v1/preflight.json').read_text())
        paths.update(ROOT/name for name in preflight['source_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    config_sha = {}
    for group, config in configs.items():
        path = args.output/f'{group}.yaml'
        path.write_text(OmegaConf.to_yaml(config, resolve=True))
        config_sha[group] = file_sha(path)
    checks = checker_hashes()
    for split, manifest in (('dev','dev.jsonl'), ('test','test_manifest.jsonl')):
        atomic_json(args.output/f'eval_{split}.json', dict(
            phase='FROZEN_FINAL_TEST' if split=='test' else 'FROZEN_DEV',
            checkpoint_selection='final_planned_step', manifest_sha256=file_sha(ROOT/f'data/posttraining_text_v3/{manifest}'),
            max_prompt_tokens=2048, max_response_tokens=30000, max_model_len=32768,
            temperature=1.0, temperature_by_domain={'math':.6,'if':1.0}, top_p=.95, top_k=-1,
            seed=17, dtype='bfloat16', checker_hashes=checks,
            execution_shards=4,
            math_test_samples=4, if_samples=1,
            limitations=['Single training seed; public benchmarks may have pretraining contamination',
                        'AIME average over four, not source paper average over 64; no direct reproduction claim']))
    protocol = dict(phase='FROZEN_TEXT_FORMAL', updates=100, save_freq=10, keep_full_checkpoints=2,
        asset_manifest_sha256=file_sha(ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json'),
        config_sha256=config_sha, evidence_sha256=evidence,
        source_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths)},
        contracts={g:replay_contract(configs[g]) for g in ('S2','S3')},
        s1_contract=sft_contract(sft), s1_budget=budget,
        eval_protocol_sha256={split:file_sha(args.output/f'eval_{split}.json') for split in ('dev','test')},
        method='Sampled-token k1 PG OPD; S3 adds independently token-normalized fixed CE, alpha=0.1',
        teacher_policy='Published source teachers; no local superiority gate',
        explicit_runtime_flags='Prefix caching/chunked prefill=True matches measured profiles; weights sync resets prefix cache',
        schedule=['S0 dev','S2 first10/restore10/finish100','S3 first10/restore10/finish100','S1 token-budget SFT/restore','S0/S1/S2/S3 test'],
        stop_conditions=['nonfinite loss/gradient','missing real update','OOM','checkpoint checksum/restore failure','insufficient disk'],
        checkpoint_selection='Final planned step only; no choosing by test score',
        estimated_training_hours={g:cost[g]['mean_step_seconds']*100/3600 for g in ('S2','S3')},
        estimate_limits='Excludes initialization, checkpoint IO, S1 and evaluation; generated length may change')
    atomic_json(args.output/'protocol.json', protocol)
    print(json.dumps({'phase':protocol['phase'], 's1_budget':budget, 'hours':protocol['estimated_training_hours']}, indent=2))


if __name__ == '__main__':
    main()
