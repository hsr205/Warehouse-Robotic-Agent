import torch
from torch import nn, Tensor


class ActorNetwork(nn.Module):
    def __init__(self, output_dimensions: int, device):
        super(ActorNetwork, self).__init__()

        # NOTE: We leverage a convolutional neural network because our observations are
        #       actually pixels that represent our grid environment
        self._convolutional_neural_network = nn.Sequential(

            # NOTE: in_channels is required to be 3 because RGB only has 3 channels
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self._fully_connected_neural_network = nn.Sequential(
            # NOTE: in_features is a product of the out_channels of the
            #       previous CNN layer and the number of discrete actions
            nn.Linear(in_features=32 * 7 * 7, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=output_dimensions),
        )

        self._device = device
        self.to(self._device)

    def forward(self, observation_tensor: Tensor) -> Tensor:

        observation_tensor = observation_tensor.to(self._device, dtype=torch.float32)

        # NOTE: For this normalization, the value, 255.0, is
        #       leveraged as that is the maximum value that a pixel can hold
        observation_tensor = observation_tensor / 255.0

        # NOTE: The shape must be changed from (Height, Width, Channel) → (Channel, Height, Width)
        if observation_tensor.dim() == 3:
            # NOTE: permute reorders the dimensions of a tensor
            observation_tensor = observation_tensor.permute(2, 0, 1)

        if observation_tensor.dim() == 3:
            # NOTE: unsqueeze adds a dimension to a tensor
            observation_tensor = observation_tensor.unsqueeze(0)

        x = self._convolutional_neural_network(observation_tensor)

        x_flatten = torch.flatten(x, start_dim=1)

        logits = self._fully_connected_neural_network(x_flatten)

        action_probabilities_tensor: Tensor = torch.softmax(logits, dim=-1)

        return action_probabilities_tensor
