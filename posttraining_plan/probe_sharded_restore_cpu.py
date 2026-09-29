"""Tiny two-rank CPU/FSDP restore exercise; not production GPU training."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]


def worker(rank, stage, output, rendezvous):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, world_size=2, rank=rank)
    try:
        torch.manual_seed(17 if stage == 'save' else 999)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        model = FSDP(LlamaForCausalLM(config), device_id=torch.device('cpu'), use_orig_params=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        manager = FSDPCheckpointManager(model, optimizer, scheduler)
        checkpoint = Path(output)/'actor'
        if stage == 'save':
            inputs = torch.tensor([[1, 2+rank, 3, 4]])
            model(input_ids=inputs, labels=inputs).loss.backward()
            optimizer.step()
            scheduler.step()
            manager.save_checkpoint(str(checkpoint), global_step=1)
        else:
            os.environ['OPD_TEXT_RESTORE_AUDIT_DIR'] = str(Path(output)/'restore_audit')
            manager.load_checkpoint(str(checkpoint), del_local_after_load=False)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('save','load'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES empty for this CPU test')
    args.output.mkdir(parents=True, exist_ok=args.stage=='load')
    import torch.multiprocessing as mp
    from verifiable_opd.posttraining.checkpoint_protocol import atomic_json
    with tempfile.TemporaryDirectory(prefix='opd-sharded-cpu-') as tmp:
        mp.spawn(worker, args=(args.stage, str(args.output.resolve()), str(Path(tmp)/'group')), nprocs=2, join=True)
    if args.stage=='load':
        reports = [json.loads((args.output/f'restore_audit/rank_{rank}.json').read_text()) for rank in (0,1)]
        if not all(row['phase']=='LOCAL_STATES_RESTORED' for row in reports):
            raise RuntimeError('Sharded state restore failed')
        atomic_json(args.output/'result.json', {'phase':'TINY_CPU_TWO_RANK_FSDP_RESTORE_PASS',
            'reports':reports, 'limits':['Tiny random model on CPU/Gloo, not 3B/NCCL/production training']})


if __name__=='__main__':
    main()
