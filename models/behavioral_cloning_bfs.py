import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import utils


class BehavioralCloningBFS(nn.Module):
    """BC with neural nets."""

    def __init__(self,
                 maze_size,
                 num_actions=4,
                 encoder_size='default',
                 augment=False,
                 encode_dim=256,
                 learning_rate=1e-3):
        super(BehavioralCloningBFS, self).__init__()

        self._maze_size = maze_size
        self._num_actions = num_actions
        self._encode_dim = encode_dim
        self._augment = augment

        if encoder_size == 'default':
            kernel_sizes = (3,) * 5
            stride_sizes = (1,) * 5
            pool_sizes = None
            num_filters = (encode_dim // 2,) * 2 + (encode_dim,) * 2 + (
                self._num_actions + 1,)
        else:
            raise NotImplementedError

        self._encoder = utils.create_conv(
            [self._maze_size, self._maze_size, 8],  # (wall; goal; loc; a)
            kernel_sizes=kernel_sizes,
            stride_sizes=stride_sizes,
            pool_sizes=pool_sizes,
            num_filters=num_filters,
            activation_last_layer=False,
        )

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
        states = torch.cat([maze_maps, loc.unsqueeze(-1)], dim=-1).float()
        if self._augment and training:
            states = self._augment_layers(states)
        return states

    def forward(self, data_batch, training=True, generate=False, device='cuda'):
        observations, maze_maps, bfs_input_maps, bfs_output_maps = data_batch
        # observations = observations#.to(device)
        # bfs_input_maps = bfs_input_maps#.to(device)
        # bfs_output_maps = bfs_output_maps#.to(device)
        maze_maps = maze_maps.int()#.to(device)

        self._optimizer.zero_grad()

        # Potential data augmentation for bfs maps
        bfs_maze_maps = torch.cat(
            [bfs_input_maps[..., None], bfs_output_maps[..., None], maze_maps],
            dim=-1) # (32, 16, 16, 4)
        bfs_input_maps, bfs_output_maps, states = torch.split(
            self.process_states(observations, bfs_maze_maps, training=training),
            [1, 1, 3],
            dim=-1)
        bfs_input_maps = bfs_input_maps[..., 0].int() # (32, 16, 16)
        bfs_output_maps = bfs_output_maps[..., 0].int() # (32, 16, 16)
        bfs_input_onehot = F.one_hot(
            bfs_input_maps.to(torch.int64), self._num_actions + 1).float() # (32, 16, 16, 5)
        bfs_states = torch.cat([states, bfs_input_onehot], dim=-1).permute(0, 3, 1, 2)  # (32, 16, 16, 8)
        logits = self._encoder(bfs_states) # (32, 5, 16, 16)
        logits = logits.reshape((-1, self._num_actions + 1)) # (8192, 5)
        labels = bfs_output_maps.view(-1).long() # (8192,)

        pred_loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        loss = torch.mean(pred_loss)

        if training:
            loss.backward()
            self._optimizer.step()

        info_dict = {
            'loss': loss.item(),
            'acc': acc.item(),
        }
        return info_dict

    def act(self, observation, maze_map, max_len=100, device='cuda'):
        maze_map = maze_map.astype(np.int)
        observations = torch.tensor([observation], dtype=torch.float32)#.to(device)
        maze_maps = torch.tensor([maze_map], dtype=torch.float32)#.to(device)
        observation = torch.tensor(observation, dtype=torch.int32)

        states = self.process_states(observations, maze_maps)
        bfs_input_maps = self._num_actions * torch.ones(
            [1, self._maze_size, self._maze_size], dtype=torch.int32)#.to(device)

        i = 0
        while bfs_input_maps[0, observation[0], observation[1]] == self._num_actions and i < max_len:
            bfs_input_onehot = F.one_hot(
                bfs_input_maps.to(torch.int64), self._num_actions + 1).float()
            bfs_states = torch.cat([states, bfs_input_onehot], dim=-1).permute(0, 3, 1, 2)
            logits = self._encoder(bfs_states)
            logits = logits.reshape((
                -1, self._maze_size, self._maze_size, self._num_actions + 1))
            bfs_input_maps = torch.argmax(
                logits, dim=-1)
            i += 1
        action = bfs_input_maps[0, observation[0], observation[1]]
        if action == self._num_actions:
            action = torch.randint(0, self._num_actions, (1,)).item()
        return action#.cpu().numpy()