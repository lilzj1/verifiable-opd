import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from posttraining_plan.sft_resources import resource_budget, init_sft_ray


class SFTResourcesTests(unittest.TestCase):
    def test_budget_fits_upstream_default(self):
        from verl.single_controller.ray import RayResourcePool
        pool = RayResourcePool(process_on_nodes=[2])
        self.assertEqual(pool.max_colocate_count, 10)
        self.assertEqual(resource_budget(2, 1, pool.max_colocate_count, 128), 24)

    def test_insufficient_host_fails_without_waiting(self):
        with self.assertRaises(ValueError):
            resource_budget(2, 1, 10, 16)
        self.assertEqual(resource_budget(2, 1, 10, 20), 20)

    def test_registered_resources_checked(self):
        config = SimpleNamespace(trainer=SimpleNamespace(n_gpus_per_node=2, nnodes=1))
        with patch('ray.init') as init, patch('ray.cluster_resources', return_value={'CPU': 24, 'GPU': 2}), \
                patch('os.sched_getaffinity', return_value=set(range(128))):
            init_sft_ray(config, {})
            self.assertEqual(init.call_args.kwargs['num_cpus'], 24)
        with patch('ray.init'), patch('ray.cluster_resources', return_value={'CPU': 16, 'GPU': 2}):
            with self.assertRaises(RuntimeError):
                init_sft_ray(config, {})


if __name__ == '__main__':
    unittest.main()
