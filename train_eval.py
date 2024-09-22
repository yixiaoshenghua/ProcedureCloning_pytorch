import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import tqdm
import dice_rl.environments.gridworld.maze as maze
from dataset import load_datasets, load_2d_datasets
from models.behavioral_cloning import BehavioralCloning
from models.behavioral_cloning_bfs import BehavioralCloningBFS
import warnings
warnings.filterwarnings("ignore")

def get_env(env_name, env_seed):
    if 'maze:' in env_name:
        # Format is in maze:<size>-<type>
        name, wall_type = env_name.split('-')
        size = int(name.split(':')[-1])
        env = maze.Maze(size, wall_type, maze_seed=env_seed)
    else:
        raise ValueError('Unknown environment: %s.' % env_name)
    return env

def get_batch(dataiter, dataset):
    try:
        observations, maze_maps, bfs_input_maps, bfs_output_maps = next(dataiter)
        batch = observations, maze_maps, bfs_input_maps, bfs_output_maps
    except StopIteration:
        dataiter = iter(dataset)
        observations, maze_maps, bfs_input_maps, bfs_output_maps = next(dataiter)
        batch = observations, maze_maps, bfs_input_maps, bfs_output_maps
    return dataiter, batch

def evaluate(env, policy, device='cuda'):
    maze_map = env.get_maze_map(stacked=True)

    total_returns = 0.0
    for i in range(args.num_eval_episodes):
        obs = env.reset()
        for j in range(args.max_eval_episode_length):
            action = policy.act(obs, maze_map, device=device)
            obs, reward, done, _ = env.step(action)
            total_returns += reward
            if done:
                break
    return total_returns / args.num_eval_episodes


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hparam_dict = {
        'env_name': args.env_name,
        'train_seeds': args.train_seeds,
        'test_seeds': args.test_seeds,
        'num_trajectory': args.num_trajectory,
        'algo_name': args.algo_name,
    }
    hparam_str = ','.join(
        ['%s=%s' % (k, str(hparam_dict[k])) for k in sorted(hparam_dict.keys())])
    summary_writer = SummaryWriter(os.path.join(args.save_dir, hparam_str))
    with open(os.path.join(os.path.join(args.save_dir, hparam_str), 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    all_envs = [get_env(args.env_name, seed) for seed in range(
        args.train_seeds + args.test_seeds)]

    if args.algo_name == 'bc':
        train_dataset, test_dataset, max_len, _ = load_datasets(
            args.load_dir,
            args.train_seeds,
            args.test_seeds,
            args.batch_size,
            args.env_name,
            args.num_trajectory,
            args.max_trajectory_length,
            build_value_map=True,
            build_bfs_sequence=False)
        algo = BehavioralCloning(
            all_envs[0].size, learning_rate=args.learning_rate, augment=False)
    elif args.algo_name == 'aug_bc':
        train_dataset, test_dataset, max_len, _ = load_datasets(
            args.load_dir,
            args.train_seeds,
            args.test_seeds,
            args.batch_size,
            args.env_name,
            args.num_trajectory,
            args.max_trajectory_length,
            build_value_map=True,
            build_bfs_sequence=False)
        algo = BehavioralCloning(
            all_envs[0].size, learning_rate=args.learning_rate, augment=True)
    elif args.algo_name == 'pc':
        train_dataset, test_dataset = load_2d_datasets(
            args.load_dir,
            args.train_seeds,
            args.test_seeds,
            args.batch_size,
            args.env_name,
            args.num_trajectory,
            args.max_trajectory_length,
            full_sequence=True)
        algo = BehavioralCloningBFS(
            all_envs[0].size,
            all_envs[0].n_action,
            learning_rate=args.learning_rate)
    elif args.algo_name == 'aux_bc':
        train_dataset, test_dataset = load_2d_datasets(
            args.load_dir,
            args.train_seeds,
            args.test_seeds,
            args.batch_size,
            args.env_name,
            args.num_trajectory,
            args.max_trajectory_length,
            full_sequence=False)
        algo = BehavioralCloningBFS(
            all_envs[0].size,
            all_envs[0].n_action,
            # aux_weight=1.,
            learning_rate=args.learning_rate)
    else:
        raise NotImplementedError

    # algo = algo.to(args.device)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    train_iter = iter(train_dataset)
    test_iter = iter(test_dataset)

    step = 0
    for step in tqdm.tqdm(range(args.num_steps)):
        train_iter, train_batch = get_batch(train_iter, train_dataset)
        info_dict = algo(train_batch, training=True, device=args.device)

        if step % args.eval_interval == 0:
            train_iter, train_batch = get_batch(train_iter, train_dataset)
            info_dict = algo(train_batch, training=True, generate=True, device=args.device)
            for k, v in info_dict.items():
                summary_writer.add_scalar(f'train/{k}', v, step)
                print('train', k, v)

            test_iter, test_batch = get_batch(test_iter, test_dataset)
            info_dict = algo(test_batch, training=False, generate=True, device=args.device)
            for k, v in info_dict.items():
                summary_writer.add_scalar(f'eval/{k}', v, step)
                print('eval', k, v)

            train_success = evaluate(all_envs[0], algo, device=args.device)
            test_successes = []
            for seed in range(args.train_seeds, args.train_seeds + args.test_seeds):
                ret = evaluate(all_envs[seed], algo, device=args.device)
                test_successes.append(ret)
            
            summary_writer.add_scalar('train/success_mean', train_success, step)
            summary_writer.add_scalar('eval/success_mean', np.mean(test_successes), step)
            summary_writer.add_scalar('eval/success_std', np.std(test_successes), step)
            summary_writer.add_scalar('eval/success_max', np.max(test_successes), step)
            summary_writer.add_scalar('eval/success_min', np.min(test_successes), step)
            print('train/success', train_success)
            print('eval/success', np.mean(test_successes), np.std(test_successes))
        step += 1
    torch.save(algo.state_dict(), '{}/model.ckpt'.format(os.path.join(args.save_dir, hparam_str)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run procedure cloning on maze BFS.')
    parser.add_argument('--env_name', type=str, default='maze:16-tunnel', help='Environment name.')
    parser.add_argument('--train_seeds', type=int, default=5, help='Number of training tasks.')
    parser.add_argument('--test_seeds', type=int, default=1, help='Number of test tasks.')
    parser.add_argument('--num_trajectory', type=int, default=4, help='Number of trajectories to collect.')
    parser.add_argument('--max_trajectory_length', type=int, default=100, help='Cutoff trajectory at this step.')
    parser.add_argument('--alpha', type=float, default=1.0, help='How close to target policy.')
    parser.add_argument('--tabular_obs', type=bool, default=False, help='Whether to use tabular observations.')
    parser.add_argument('--load_dir', type=str, default='/tmp/procedure_cloning/', help='Directory to load dataset from.')
    parser.add_argument('--save_dir', type=str, default='./logdir', help='Directory to save result to.')
    parser.add_argument('--algo_name', type=str, default='pc', choices=['bc', 'aug_bc', 'pc', 'aux_bc'], help='Algorithm name.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--num_steps', type=int, default=500_000, help='Number of training steps.')
    parser.add_argument('--eval_interval', type=int, default=10_000, help='Number of steps between each evaluation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_eval_episodes', type=int, default=5, help='Number of eval episodes.')
    parser.add_argument('--max_eval_episode_length', type=int, default=100, help='Number of eval episodes.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    main(args)