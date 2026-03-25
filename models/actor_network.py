import numpy as np
import torch
from torch import nn, Tensor


class ActorNetwork(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int):
        super(ActorNetwork, self).__init__()

        self._actor_network = nn.Sequential(
            nn.Linear(in_features=input_dimensions, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dimensions),
        )

        self._device = self._get_device()

    def forward(self, observation) -> Tensor:

        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float, device=self._device)

        if observation.dim() == 1:
            # Add a new dimension to the observation at a specified location
            observation = observation.unsqueeze(0)

        # Once the observation is valid,
        # Pass the observation through the network and return raw logits
        logits = self._actor_network(observation)

        # Using SoftMax, convert the logits into action probabilities
        action_probabilities_tensor: Tensor = torch.softmax(logits, dim=-1)

        return action_probabilities_tensor

    def _get_device(self):

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
