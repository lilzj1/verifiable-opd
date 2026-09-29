"""S3-only process-local adapter; original sources/configs remain frozen."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.rotation_handoff import verify_amendment, enable_single_rotation, AMENDMENT


def install_runtime_hook(config):
    from omegaconf import OmegaConf
    from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha
    import os
    expected = ROOT/'artifacts/text_s3_formal_s100_seed17_v1/checkpoints'
    if Path(config.trainer.default_local_dir) != expected:
        raise ValueError('Only S3 may use the amended launcher')
    OmegaConf.update(config, 'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook',
                    'posttraining_plan.rotation_handoff.worker_setup', force_add=True)
    stage = Path(os.environ['OPD_TEXT_UPDATE_AUDIT_DIR']).parent
    atomic_json(stage/'io_amendment.json', {'amendment_sha256': file_sha(AMENDMENT),
                'keep': 1, 'runtime_hook': 'posttraining_plan.rotation_handoff.worker_setup'})
    (stage/'resolved_with_io_hook.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))


def main():
    verify_amendment()
    from posttraining_plan import run_text_formal as original
    from verl.trainer import main_ppo
    original_run = main_ppo.run_ppo

    def run_with_hook(config, *args, **kwargs):
        install_runtime_hook(config)
        return original_run(config, *args, **kwargs)

    original.enable_automatic_rotation = enable_single_rotation
    main_ppo.run_ppo = run_with_hook
    original.main()


if __name__ == '__main__':
    main()
