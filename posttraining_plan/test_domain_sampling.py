import unittest
from verifiable_opd.posttraining.domain_sampling import sampling_for_domain


class DomainSamplingTests(unittest.TestCase):
    def test_domain_budgets_preserve_sampling(self):
        params = {'temperature':1., 'max_new_tokens':99, 'top_k':-1}
        for domain, n in {'math':16384, 'if':2048}.items():
            result = sampling_for_domain(params, domain, {'math':16384,'if':2048}, 16384)
            self.assertEqual(result['max_tokens'], n)
            self.assertNotIn('max_new_tokens', result)
            self.assertEqual(result['temperature'], 1.)
        self.assertEqual(params['max_new_tokens'], 99)

    def test_invalid_domain_and_budget_fail(self):
        with self.assertRaises(ValueError):
            sampling_for_domain({}, 'unknown', {'math':16384}, 16384)
        with self.assertRaises(ValueError):
            sampling_for_domain({}, 'math', {'math':16384}, 8192)

    def test_legacy_is_unchanged(self):
        self.assertEqual(sampling_for_domain({'max_tokens':20}, None, {}, 100), {'max_tokens':20})


if __name__ == '__main__':
    unittest.main()
