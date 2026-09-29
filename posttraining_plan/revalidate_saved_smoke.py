"""Recover saved state after diagnosed post-save failures, without repeating training."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from omegaconf import OmegaConf
from posttraining_plan.run_text_smoke import audit_updates
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, replay_contract, verify_before_restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--group', required=True, choices=('S2','S3'))
    args = parser.parse_args()
    if not args.run_id.replace('_','').isalnum():
        raise ValueError('Safe run ID required')
    run = ROOT/'artifacts'/args.run_id
    output = run/'smoke_revalidated.json'
    if output.exists():
        raise ValueError('Revalidation already exists; preserve evidence')
    old = json.loads((run/'smoke_result.json').read_text())
    error = old.get('error', '')
    schedule_failure = error == 'ValueError: Resume world size/data/replay contract changed'
    sync_failure = (args.group == 'S3' and args.run_id == 'text_s3_source16k_smoke_v1'
                    and 'Call to wake_up method failed' in error
                    and 'out of memory at /workspace/csrc/cumem_allocator.cpp:62' in error)
    if old.get('phase') != 'SMOKE_FAILED' or not (schedule_failure or sync_failure):
        raise ValueError('Only the diagnosed post-save schedule or source16k S3 weight-sync failures are supported')
    cfg = OmegaConf.load(run/'resolved.yaml')
    loss = cfg.distillation.distillation_loss
    if (args.group == 'S2' and loss.expert_replay_strategy != 'none') or (args.group == 'S3' and loss.expert_replay_strategy != 'uniform'):
        raise ValueError('Group mismatch')
    checkpoint = run/'checkpoints/global_step_1'
    commit = json.loads((checkpoint/'TEXT_COMMIT.json').read_text())
    if schedule_failure and (cfg.actor_rollout_ref.actor.optim.total_training_steps != -1 or commit['contract']['optimizer']['total_training_steps'] != cfg.trainer.total_training_steps):
        raise ValueError('Not the diagnosed unset-to-effective schedule difference')
    records = audit_updates(run/'optimizer_audit')
    started = time.monotonic()
    print('Revalidating all saved shard hashes; no GPU or training.', flush=True)
    manifest = verify_before_restore(checkpoint, 2, replay_contract(cfg))
    atomic_json(output, {
        'phase':'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING', 'group':args.group,
        'optimizer_audit':records, 'checkpoint_commit_sha256':file_sha(checkpoint/'TEXT_COMMIT.json'),
        'checkpoint_bytes':sum(entry['bytes'] for entry in manifest['files'].values()),
        'original_failure_sha256':file_sha(run/'smoke_result.json'),
        'original_config_sha256':file_sha(run/'resolved.yaml'),
        'revalidation_seconds':time.monotonic()-started, 'allow_formal':False,
        'repair':('Canonicalize optimizer total_training_steps=-1; other fields and shard hashes unchanged' if schedule_failure
                  else 'Preserve successful update/save despite later vLLM wake-up OOM; actual restore and subsequent weight sync remain to verify'),
        'post_update_weight_sync_passed': not sync_failure,
    })
    print('Saved evidence:', output, flush=True)


if __name__ == '__main__':
    main()
