"""Match the upstream SFT placement group before launching any workers."""
import os


def resource_budget(n_gpus, nnodes, cpus_per_gpu, host_cpus):
    if nnodes != 1 or n_gpus != 2:
        raise ValueError('This SFT entry point is registered for one node / two GPUs')
    required = n_gpus * cpus_per_gpu
    if host_cpus < required:
        raise ValueError(f'SFT placement needs {required} CPUs; host exposes {host_cpus}')
    return min(required + 4, host_cpus)


def init_sft_ray(config, env):
    import ray
    from verl.single_controller.ray import RayResourcePool

    # SFTTrainer constructs this same pool with its upstream defaults. Do not
    # hard-code a smaller CPU budget than the resulting placement group.
    pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    host_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    required = config.trainer.n_gpus_per_node * pool.max_colocate_count
    cpus = resource_budget(config.trainer.n_gpus_per_node, config.trainer.nnodes,
                           pool.max_colocate_count, host_cpus)
    ray.init(address='local', num_cpus=cpus, num_gpus=2, include_dashboard=False,
             runtime_env={'env_vars': env})
    resources = ray.cluster_resources()
    if resources.get('CPU', 0) < required or resources.get('GPU', 0) < 2:
        raise RuntimeError(f'SFT placement cannot fit in private Ray cluster: {resources}')
    print(f'SFT_RESOURCE_PREFLIGHT required_cpus={required} registered_cpus={cpus} gpus=2', flush=True)
