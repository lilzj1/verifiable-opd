"""One-GPU tiny random-model test of the REAL FSDP checkpoint manager hooks.

Not a published-model update, not a two-rank test, and not an OPD result.
Run save/load as separate processes to avoid an in-memory pseudo-restore.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'upstream/verl-v0.8.0'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('save', 'load'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4':
        raise ValueError('This diagnostic is restricted to physical GPU4')
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from verifiable_opd.posttraining.checkpoint_protocol import atomic_json
    args.output.mkdir(parents=True, exist_ok=args.stage == 'load')
    with tempfile.TemporaryDirectory(prefix='opd-restore-rendezvous-') as tmp:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group('nccl', init_method='file://' + tmp + '/group', rank=0, world_size=1)
        try:
            torch.manual_seed(17 if args.stage == 'save' else 999)
            config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
            model = FSDP(LlamaForCausalLM(config), device_id=0, use_orig_params=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
            manager = FSDPCheckpointManager(model, optimizer, scheduler)
            checkpoint = args.output / 'actor'
            if args.stage == 'save':
                inputs = torch.tensor([[1, 2, 3, 4]], device='cuda')
                loss = model(input_ids=inputs, labels=inputs).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                manager.save_checkpoint(str(checkpoint), global_step=1)
                atomic_json(args.output / 'save_result.json', {
                    'phase': 'TINY_ONE_GPU_SAVE_COMPLETE', 'loss': loss.item(),
                    'scope': 'Random tiny Llama, not Math/IF models or OPD training'})
            else:
                os.environ['OPD_TEXT_RESTORE_AUDIT_DIR'] = str(args.output / 'restore_audit')
                manager.load_checkpoint(str(checkpoint), del_local_after_load=False)
                report = json.loads((args.output / 'restore_audit/rank_0.json').read_text())
                if report['phase'] != 'LOCAL_STATES_RESTORED':
                    raise RuntimeError('Tiny model restore did not match')
                atomic_json(args.output / 'load_result.json', {
                    'phase': 'TINY_ONE_GPU_FSDP_RESTORE_PASS', 'audit': report,
                    'scope': 'Separate-process diagnostic only; full-model two-rank restore still pending'})
                print('TINY_ONE_GPU_FSDP_RESTORE_PASS', flush=True)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
