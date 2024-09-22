import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import dummy as multiprocessing
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dice_rl.data.dataset import Dataset
import dice_rl.environments.gridworld.maze as maze

import utils


def load_datasets(load_dir,
                  train_seeds,
                  test_seeds,
                  batch_size,
                  env_name,
                  num_trajectory,
                  max_trajectory_length,
                  stacked=True,
                  build_value_map=False,
                  build_bfs_sequence=False):
    pool = multiprocessing.Pool(100)

    def load_dataset_env(seed):
        name, wall_type = env_name.split('-')
        size = int(name.split(':')[-1])
        env = maze.Maze(size, wall_type, maze_seed=seed)
        hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                      'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                          ENV_NAME=env_name,
                          TAB=False,
                          ALPHA=1.0,
                          SEED=seed,
                          NUM_TRAJ=num_trajectory,
                          MAX_TRAJ=max_trajectory_length)
        directory = os.path.join(load_dir, hparam_str)
        dataset = Dataset.load(directory)
        return dataset, env

    datasets_envs = pool.map(load_dataset_env, range(train_seeds + test_seeds))

    observations = []
    actions = []
    maze_maps = []
    value_maps = []
    bfs_sequences = []
    max_len = 0
    max_bfs_len = 0
    for (dataset, env) in datasets_envs:
        episodes, valid_steps = dataset.get_all_episodes()
        max_len = max(max_len, valid_steps.shape[1])

        env_steps = dataset.get_all_steps(num_steps=1)
        observation = torch.squeeze(torch.tensor(env_steps.observation.numpy()), dim=1)
        action = torch.squeeze(torch.tensor(env_steps.action.numpy(), dtype=torch.int32), dim=1)

        observations.append(observation)
        actions.append(action)
        maze_map = env.get_maze_map(stacked=stacked)
        maze_maps.append(
            torch.repeat_interleave(torch.tensor(maze_map)[None, ...], env_steps.observation.shape[0], dim=0))

        value_map = torch.tensor(maze.get_value_map(env), dtype=torch.float32)
        value_maps.append(
            torch.repeat_interleave(value_map[None, ...], env_steps.observation.shape[0], dim=0))

        bfs_sequence = []
        for i in range(observation.shape[0]):
            bfs_sequence_single = maze.get_bfs_sequence(
                env, observation[i].numpy().astype(int), include_maze_layout=True)
            max_bfs_len = max(max_bfs_len, len(bfs_sequence_single))
            bfs_sequence.append(bfs_sequence_single)
        bfs_sequences.append(bfs_sequence)

    train_data = (torch.cat(observations[:train_seeds], dim=0), torch.cat(actions[:train_seeds], dim=0),
                  torch.cat(maze_maps[:train_seeds], dim=0))

    test_data = (torch.cat(observations[train_seeds:], dim=0), torch.cat(actions[train_seeds:], dim=0),
                 torch.cat(maze_maps[train_seeds:], dim=0))

    if build_value_map:
        train_data += (torch.cat(value_maps[:train_seeds], dim=0),)
        test_data += (torch.cat(value_maps[train_seeds:], dim=0),)
    if build_bfs_sequence:
        train_sequences = [
            seq for bfs_sequence in bfs_sequences[:train_seeds]
            for seq in bfs_sequence
        ]
        test_sequences = [
            seq for bfs_sequence in bfs_sequences[train_seeds:]
            for seq in bfs_sequence
        ]
        vocab_size = datasets_envs[0][1].n_action + datasets_envs[0][1].num_maze_keys
        train_sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in train_sequences], batch_first=True, padding_value=vocab_size)
        test_sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in test_sequences], batch_first=True, padding_value=vocab_size)
        train_data += (train_sequences,)
        test_data += (test_sequences,)

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader, max_len, max_bfs_len


def load_2d_datasets(load_dir,
                     train_seeds,
                     test_seeds,
                     batch_size,
                     env_name,
                     num_trajectory,
                     max_trajectory_length,
                     full_sequence=True):
    pool = multiprocessing.Pool(100)

    def load_dataset_env(seed):
        name, wall_type = env_name.split('-')
        size = int(name.split(':')[-1])
        env = maze.Maze(size, wall_type, maze_seed=seed)
        hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                      'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                          ENV_NAME=env_name,
                          TAB=False,
                          ALPHA=1.0,
                          SEED=seed,
                          NUM_TRAJ=num_trajectory,
                          MAX_TRAJ=max_trajectory_length)
        directory = os.path.join(load_dir, hparam_str)
        dataset = Dataset.load(directory)
        return dataset, env

    datasets_envs = pool.map(load_dataset_env, range(train_seeds + test_seeds))

    observations_train = []
    observations_test = []
    maze_maps_train = []
    maze_maps_test = []
    bfs_input_maps_train = []
    bfs_input_maps_test = []
    bfs_output_maps_train = []
    bfs_output_maps_test = []
    for idx, (dataset, env) in enumerate(datasets_envs):
        if idx < train_seeds:
            observations = observations_train
            maze_maps = maze_maps_train
            bfs_input_maps = bfs_input_maps_train
            bfs_output_maps = bfs_output_maps_train
        else:
            observations = observations_test
            maze_maps = maze_maps_test
            bfs_input_maps = bfs_input_maps_test
            bfs_output_maps = bfs_output_maps_test

        episodes, valid_steps = dataset.get_all_episodes()
        env_steps = dataset.get_all_steps(num_steps=1)
        env_observations = torch.squeeze(torch.tensor(env_steps.observation.numpy()), dim=1)
        maze_map = env.get_maze_map(stacked=True)
        for i in range(env_observations.shape[0]):
            bfs_sequence = utils.get_vi_sequence(
                env, env_observations[i].numpy().astype(np.int32))  # [L, W, W]
            bfs_input_map = env.n_action * torch.ones([env.size, env.size], dtype=torch.int32)
            if full_sequence:
                for j in range(bfs_sequence.shape[0]):
                    bfs_input_maps.append(bfs_input_map)
                    bfs_output_maps.append(torch.tensor(bfs_sequence[j]))
                    observations.append(env_observations[i])
                    maze_maps.append(torch.tensor(maze_map))
                    bfs_input_map = torch.tensor(bfs_sequence[j])
            else:
                bfs_input_maps.append(bfs_input_map)
                bfs_output_maps.append(torch.tensor(bfs_sequence[-1]))
                observations.append(env_observations[i])
                maze_maps.append(torch.tensor(maze_map))

    train_data = (
        torch.stack(observations_train, dim=0),
        torch.stack(maze_maps_train, dim=0),
        torch.stack(bfs_input_maps_train, dim=0),
        torch.stack(bfs_output_maps_train, dim=0),
    )
    test_data = (
        torch.stack(observations_test, dim=0),
        torch.stack(maze_maps_test, dim=0),
        torch.stack(bfs_input_maps_test, dim=0),
        torch.stack(bfs_output_maps_test, dim=0),
    )

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader