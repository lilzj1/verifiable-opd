"""Merge a completed, verified text checkpoint to HF on CPU. Never upload."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.run_text_formal import read_protocol
from posttraining_plan.run_text_smoke import PYTHON
from verifiable_opd.posttraining.checkpoint_protocol import verify_before_restore, file_sha, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--group', choices=['S1','S2','S3'], required=True)
    args = parser.parse_args()
    protocol, config = read_protocol(args.protocol, args.group)
    run = Path(config.trainer.default_local_dir).parent
    if args.group == 'S1':
        result_path, phase = run/'train/result.json', 'SFT_FORMAL_SAVED_EVALUATION_PENDING'
        contract, model = protocol['s1_contract'], config.model.path
        if json.loads((run/'restore/restore_result.json').read_text())['phase'] != 'DISTRIBUTED_RESTORE_VERIFIED':
            raise ValueError('S1 final restore required')
    else:
        result_path, phase = run/'stages/finish/result.json', 'HUNDRED_UPDATES_COMPLETE_EVALUATION_PENDING'
        contract, model = protocol['contracts'][args.group], config.actor_rollout_ref.model.path
    result = json.loads(result_path.read_text())
    if result['phase'] != phase:
        raise ValueError('No complete final checkpoint to export')
    checkpoint = run/f'checkpoints/global_step_{config.trainer.total_training_steps}'
    verify_before_restore(checkpoint, 2, contract)
    if file_sha(checkpoint/'TEXT_COMMIT.json') != result['checkpoint_commit_sha256']:
        raise ValueError('Final commit changed')
    target = run/'final_hf'
    target.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}')
    subprocess.run([PYTHON, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                    '--local_dir', str(checkpoint/'actor'), '--target_dir', str(target)], env=env, check=True)
    from transformers import AutoTokenizer, AutoConfig
    from safetensors import safe_open
    old_tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, fix_mistral_regex=False)
    new_tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, fix_mistral_regex=False)
    if old_tokenizer.backend_tokenizer.to_str() != new_tokenizer.backend_tokenizer.to_str() or old_tokenizer.chat_template != new_tokenizer.chat_template:
        raise ValueError('Export changed tokenizer or chat template')
    original, merged = AutoConfig.from_pretrained(model), AutoConfig.from_pretrained(target)
    for key in ('model_type','vocab_size','hidden_size','num_hidden_layers','rope_theta','rope_scaling'):
        if getattr(original,key,None) != getattr(merged,key,None):
            raise ValueError(f'Export changed model setting: {key}')
    files = list(target.glob('*.safetensors'))
    if not files:
        raise ValueError('No HF weights exported')
    keys = set()
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in keys:
                    raise ValueError('Duplicate exported tensor')
                keys.add(key)
                if not stream.get_slice(key).get_shape():
                    raise ValueError('Unexpected scalar model weight')
    index = target/'model.safetensors.index.json'
    if index.exists() and set(json.loads(index.read_text())['weight_map']) != keys:
        raise ValueError('Exported shard index is incomplete')
    atomic_json(run/'export_result.json', dict(phase='FINAL_HF_EXPORTED', group=args.group,
        checkpoint=str(checkpoint), checkpoint_commit_sha256=file_sha(checkpoint/'TEXT_COMMIT.json'),
        target=str(target), tensor_count=len(keys),
        files={p.name:file_sha(p) for p in target.iterdir() if p.is_file()},
        limitation='Verified structure/tokenizer/config and checksums; downstream inference still required'))


if __name__ == '__main__':
    main()
