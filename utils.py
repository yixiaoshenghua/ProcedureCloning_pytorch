"""Utilities."""

import numpy as np
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: typing.Sequence[int] = (256, 256),
    activation: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU(),
    near_zero_last_layer: bool = False,
) -> nn.Sequential:
    """Creates an MLP.

    Args:
        input_dim: input dimensionaloty.
        output_dim: output dimensionality.
        hidden_dims: hidden layers dimensionality.
        activation: activations after hidden units.

    Returns:
        An MLP model.
    """
    initialization = nn.init.kaiming_uniform_
    near_zero_initialization = nn.init.normal_

    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        input_dim = hidden_dim

    layers.append(nn.Linear(input_dim, output_dim))
    if near_zero_last_layer:
        nn.init.normal_(layers[-1].weight, std=1e-2)
        nn.init.zeros_(layers[-1].bias)
    else:
        initialization(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    return nn.Sequential(*layers)

def same_padding(input_size, kernel_size, stride):
    pad_total = max((math.ceil(input_size / stride) - 1) * stride + kernel_size - input_size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after

def create_conv(
    input_shape: typing.Sequence[typing.Optional[int]],
    kernel_sizes: typing.Sequence[int] = (2, 2, 3),
    stride_sizes: typing.Sequence[int] = (1, 1, 2),
    pool_sizes: typing.Optional[typing.Sequence[typing.Optional[int]]] = None,
    num_filters: typing.Union[int, typing.Sequence[int]] = 64,
    activation: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU(),
    activation_last_layer: bool = True,
    output_dim: typing.Optional[int] = None,
    padding: str = 'same',
    residual: bool = False,
) -> nn.Sequential:
    """Creates a Convolutional Neural Network.

    Args:
        input_shape: input shape.
        kernel_sizes: kernel sizes for each layer.
        stride_sizes: stride sizes for each layer.
        pool_sizes: pool sizes for each layer.
        num_filters: number of filters for each layer.
        activation: activation function after hidden units.
        activation_last_layer: whether to apply activation to the last layer.
        output_dim: output dimensionality.
        padding: padding mode.
        residual: whether to use residual connections.

    Returns:
        A CNN model.
    """
    if not hasattr(num_filters, '__len__'):
        num_filters = (num_filters,) * len(kernel_sizes)
    if not hasattr(pool_sizes, '__len__'):
        pool_sizes = (pool_sizes,) * len(kernel_sizes)

    layers = []
    in_channels = input_shape[2]
    for i, (kernel_size, stride, filters, pool_size) in enumerate(
            zip(kernel_sizes, stride_sizes, num_filters, pool_sizes)):
        if i == len(kernel_sizes) - 1 and not output_dim:
            activation_fn = activation if activation_last_layer else None
        else:
            activation_fn = activation

        layers.append(nn.Conv2d(in_channels, filters, kernel_size, stride, padding=padding))
        if activation_fn:
            layers.append(activation_fn)
        if pool_size:
            # pad_h = same_padding(input_shape[0], kernel_size, stride)
            # pad_w = same_padding(input_shape[1], kernel_size, stride)
            layers.append(nn.MaxPool2d(pool_size))
        in_channels = filters

    if output_dim:
        layers += [
            nn.Flatten(),
            nn.Linear(in_channels, output_dim)
        ]
        if activation_last_layer:
            layers.append(activation)

    model = nn.Sequential(*layers)
    return model


def get_vi_sequence(env, observation):
    """Returns [L, W, W] optimal actions."""
    start_x, start_y = observation
    target_location = env.target_location
    nav_map = env.nav_map
    current_points = [target_location]
    chosen_actions = {target_location: 0}
    visited_points = {target_location: True}
    vi_sequence = []
    vi_map = np.full((env.size, env.size),
                     fill_value=env.n_action,
                     dtype=np.int32)

    found_start = False
    while current_points and not found_start:
        next_points = []
        for point_x, point_y in current_points:
            for (action, (next_point_x,
                          next_point_y)) in [(0, (point_x - 1, point_y)),
                                             (1, (point_x, point_y - 1)),
                                             (2, (point_x + 1, point_y)),
                                             (3, (point_x, point_y + 1))]:

                if (next_point_x, next_point_y) in visited_points:
                    continue

                if not (next_point_x >= 0 and next_point_y >= 0 and
                        next_point_x < len(nav_map) and
                        next_point_y < len(nav_map[next_point_x])):
                    continue

                if nav_map[next_point_x][next_point_y] == 'x':
                    continue

                next_points.append((next_point_x, next_point_y))
                visited_points[(next_point_x, next_point_y)] = True
                chosen_actions[(next_point_x, next_point_y)] = action
                vi_map[next_point_x, next_point_y] = action

                if next_point_x == start_x and next_point_y == start_y:
                    found_start = True
        vi_sequence.append(vi_map.copy())
        current_points = next_points

    return np.array(vi_sequence)