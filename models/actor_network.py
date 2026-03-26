import numpy as np
import torch
from torch import nn, Tensor


class ActorNetwork(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int, device):
        super(ActorNetwork, self).__init__()

        self._actor_network = nn.Sequential(
            nn.Linear(in_features=input_dimensions, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dimensions),
        )

        self._device = device
        self.to(self._device)

    def forward(self, observation_tensor:Tensor) -> Tensor:

        if isinstance(observation_tensor, np.ndarray):
            observation_tensor = torch.tensor(observation_tensor, dtype=torch.float, device=self._device)

        print("Here")

        if observation_tensor.dim() == 1:
            # Add a new dimension to the observation at a specified location
            observation_tensor = observation_tensor.unsqueeze(0)

        print("Now Here")

        # Once the observation is valid,
        # Pass the observation through the network and return raw logits
        # print(f"observation = {observation}")
        # print(f"type(observation) = {type(observation)}")

        logits = self._actor_network(observation_tensor)

        print("After logits")

        # Using SoftMax, convert the logits into action probabilities
        action_probabilities_tensor: Tensor = torch.softmax(logits, dim=-1)

        print("End of forward() pass method")

        return action_probabilities_tensor

