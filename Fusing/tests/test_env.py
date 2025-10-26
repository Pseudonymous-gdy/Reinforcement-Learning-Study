import math
import unittest
import numpy as np

from Fusing.env import Environment


class TestEnvironment(unittest.TestCase):
    def test_same_seed_same_means_and_sequence(self):
        env1 = Environment(10, seed=123)
        env2 = Environment(10, seed=123)

        # means should be identical
        self.assertEqual(env1.get_bandit_means(), env2.get_bandit_means())

        # reward sequence for a fixed arm should be identical
        arm = 3
        seq1 = [env1.get_reward(arm) for _ in range(20)]
        seq2 = [env2.get_reward(arm) for _ in range(20)]
        self.assertEqual(seq1, seq2)

    def test_different_seed_different_means(self):
        env1 = Environment(10, seed=123)
        env2 = Environment(10, seed=124)

        # bandit parameters should differ deterministically here
        self.assertNotEqual(env1.get_bandit_means(), env2.get_bandit_means())

    def test_get_dueling_ordering_on_sorted_means(self):
        env = Environment(8, seed=7)
        means = env.get_bandit_means()

        # means are sorted descending by construction
        self.assertTrue(all(means[i] >= means[i + 1] for i in range(len(means) - 1)))

        # top arm should beat the next
        self.assertEqual(env.get_dueling(0, 1), 1)
        # lower arm should not beat higher arm
        self.assertEqual(env.get_dueling(2, 0), 0)

    def test_get_dueling_index_out_of_range(self):
        env = Environment(5, seed=0)
        with self.assertRaises(IndexError):
            env.get_dueling(-1, 0)
        with self.assertRaises(IndexError):
            env.get_dueling(0, 5)

    def test_reset_with_same_seed_repeats_sequence(self):
        env = Environment(6, seed=99)
        arm = 2
        seq1 = [env.get_reward(arm) for _ in range(15)]

        # reset RNG to same seed; bandit parameters remain identical
        env.reset(seed=99)
        seq2 = [env.get_reward(arm) for _ in range(15)]

        self.assertEqual(seq1, seq2)

    def test_normal_distribution_shapes_and_history(self):
        env = Environment(4, distribution='normal', seed=2024)
        arm = 1
        values = [env.get_reward(arm) for _ in range(50)]
        # history length matches
        self.assertEqual(len(env.get_history()), 50)
        # values are floats
        self.assertTrue(all(isinstance(v, float) for v in values))
        # sample mean should be reasonably close to the bandit's mean (within a tolerance)
        m = float(np.mean(values))
        self.assertLess(abs(m - float(env.bandit_means[arm])), 0.5)  # loose tolerance due to small sample size


if __name__ == '__main__':
    unittest.main()
