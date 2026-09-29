import unittest
from verifiable_opd.environment import ArithmeticEnv, Task, oracle_action


class EnvironmentTests(unittest.TestCase):
    def task(self):
        return Task(3, (('add', 2), ('mul', 4)))

    def test_expert_both_modes(self):
        for mode in ('sparse', 'constraint'):
            env = ArithmeticEnv(self.task(), mode)
            _, reward, done = env.step(oracle_action(env))
            self.assertEqual((reward, done), (0.0, False))
            obs, reward, done = env.step(oracle_action(env))
            self.assertEqual((obs['value'], reward, done), (20, 1.0, True))

    def test_sparse_wrong_advances_without_correctness_feedback(self):
        env = ArithmeticEnv(self.task())
        obs, reward, done = env.step('{"value":6}')
        self.assertNotIn('feedback', obs)
        self.assertEqual((obs['step'], reward, done), (1, 0.0, False))
        _, reward, done = env.step(oracle_action(env))
        self.assertEqual((reward, done), (0.0, True))

    def test_constraint_repair(self):
        env = ArithmeticEnv(self.task(), 'constraint')
        obs, _, _ = env.step('{"value":6}')
        self.assertEqual((obs['value'], obs['step']), (3, 0))
        self.assertEqual(obs['feedback'], 'rule_check_failed_retry')
        env.step(oracle_action(env))
        _, reward, done = env.step(oracle_action(env))
        self.assertEqual((reward, done), (1.0, True))

    def test_budget_termination(self):
        env = ArithmeticEnv(self.task(), 'constraint', max_calls=1)
        _, reward, done = env.step('{"value":6}')
        self.assertEqual((reward, done), (0.0, True))
        with self.assertRaises(RuntimeError):
            env.step('{"value":5}')

    def test_invalid_actions(self):
        for action in ['not json', '{"value":true}', '{"value":999}', '[]', '{"value":5,"extra":1}']:
            env = ArithmeticEnv(self.task(), max_calls=1)
            obs, reward, done = env.step(action)
            self.assertEqual(obs['feedback'], 'invalid_json_or_value')
            self.assertEqual((reward, done), (0.0, True))

    def test_reset_and_modulus(self):
        env = ArithmeticEnv(Task(9, (('mul', 3),), modulus=10))
        obs, reward, _ = env.step(oracle_action(env))
        self.assertEqual((obs['value'], reward), (7, 1.0))
        self.assertEqual(env.reset()['value'], 9)


if __name__ == '__main__':
    unittest.main()
