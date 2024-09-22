"""Makes sure that scripts/train_eval.py runs without error."""

import unittest
import torch
import train_eval


class TrainEvalTest(unittest.TestCase):

    def test_train_eval(self):
        train_eval.FLAGS.train_seeds = 5
        train_eval.FLAGS.test_seeds = 1
        train_eval.FLAGS.num_trajectory = 4
        train_eval.FLAGS.load_dir = './tests/testdata/'
        train_eval.FLAGS.algo_name = 'pc'
        train_eval.FLAGS.num_steps = 10
        train_eval.FLAGS.eval_interval = 10
        train_eval.FLAGS.batch_size = 1
        train_eval.FLAGS.num_eval_episodes = 1
        train_eval.FLAGS.max_eval_episode_length = 5
        with self.assertRaises(SystemExit):
            train_eval.main([])


if __name__ == '__main__':
    unittest.main()
