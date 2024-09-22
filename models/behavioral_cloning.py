import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import utils


class BehavioralCloning(nn.Module):
    """BC with neural nets."""

    def __init__(self,
                 maze_size,
                 num_actions=4,
                 encoder_size='default',
                 augment=True,
                 encode_dim=256,
                 aux_weight=0.,
                 learning_rate=1e-3):
        super(BehavioralCloning, self).__init__()

        self._maze_size = maze_size
        self._num_actions = num_actions
        self._encode_dim = encode_dim
        self._augment = augment
        self._aux_weight = aux_weight

        if encoder_size == 'default':
            kernel_sizes = (3,) * 5
            stride_sizes = (1,) * 5
            pool_sizes = (2, 2, 2, 2, None)
            num_filters = (encode_dim // 2,) * 2 + (encode_dim,) * 3
        else:
            raise NotImplementedError

        self._encoder = utils.create_conv(
            [self._maze_size, self._maze_size, 3],  # (wall; goal; loc)
            kernel_sizes=kernel_sizes,
            stride_sizes=stride_sizes,
            pool_sizes=pool_sizes,
            num_filters=num_filters,
            output_dim=encode_dim)

        self._policy = utils.create_mlp(encode_dim, self._num_actions)
        self._action_network = utils.create_mlp(encode_dim, maze_size * maze_size * num_actions)

        if self._augment:
            self._augment_layers = nn.Sequential(
                transforms.RandomCrop(maze_size),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
                transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=0),
            )

        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def process_states(self, observations, maze_maps, training=True):
        """Returns [B, W, W, 3] binary values. Channels are (wall; goal; obs)"""
        loc = F.one_hot(
            torch.tensor(observations[:, 0] * self._maze_size + observations[:, 1], dtype=torch.int64),
            self._maze_size * self._maze_size)
        loc = loc.view(-1, self._maze_size, self._maze_size)
        maze_maps = maze_maps.float()
        states = torch.cat([maze_maps, loc.unsqueeze(-1)], dim=-1)
        if self._augment and training:
            states = self._augment_layers(states)
        return states

    def embed_states(self, states):
        return self._encoder(states.permute(0, 3, 1, 2))

    def forward(self, data_batch, training=True, generate=False, device='cuda'):
        observations, actions, maze_maps, value_maps = data_batch
        # observations = observations.to(device)
        # bfs_input_maps = bfs_input_maps.to(device)
        # bfs_output_maps = bfs_output_maps.to(device)

        states = self.process_states(observations, maze_maps, training=training)

        self._optimizer.zero_grad()
        embed = self.embed_states(states)
        logit = self._policy(embed)
        pred_loss = F.cross_entropy(logit, actions.long())
        pred = torch.argmax(logit, dim=-1)
        acc = (pred == actions).float().mean()

        value_maps = value_maps.reshape((-1, self._num_actions))
        valid_indices = torch.where(value_maps.sum(dim=-1) == 1)[0]
        value_maps = value_maps[valid_indices]
        logit = self._action_network(embed)
        logit = logit.reshape((-1, self._num_actions))
        logit = logit[valid_indices]
        aux_loss = F.cross_entropy(logit, value_maps)

        loss = torch.mean(pred_loss) + self._aux_weight * torch.mean(aux_loss)
        if training:
            loss.backward()
            self._optimizer.step()

        return {'loss': loss.item(),
                'pred_loss': torch.mean(pred_loss).item(),
                'aux_loss': torch.mean(aux_loss).item(),
                'acc': acc.item()
                }

    def act(self, observation, maze_map, device='cuda'):
        observations = torch.tensor([observation], dtype=torch.float32)
        maze_maps = torch.tensor([maze_map], dtype=torch.float32)
        states = self.process_states(observations, maze_maps, training=False)
        embed = self.embed_states(states)
        logit = self._policy(embed)
        dist = torch.distributions.Categorical(F.softmax(logit, dim=-1))
        return dist.sample((1,)).item()