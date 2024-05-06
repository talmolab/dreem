"""Multi-Layer Perceptron (MLP) module."""

import torch


class MLP(torch.nn.Module):
    """Multi-layer perceptron class."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        """Initialize MLP.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dim: Number of units in the hidden layers.
            output_dim: Dimensionality of the output features.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if self.num_layers > 0:
            self.layers = [torch.nn.LazyLinear(hidden_dim)]
            self.dropouts = [torch.nn.Dropout(dropout)]
            for i in range(num_layers):
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.dropouts.append(torch.nn.Dropout(dropout))
            self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
            self.dropouts.append(torch.nn.Dropout(0.0))

            self.layers = torch.nn.ModuleList(self.layers)
            self.dropouts = torch.nn.ModuleList(self.dropouts)
        else:
            self.layers = torch.nn.ModuleList([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (batch_size, num_instances, input_dim).

        Returns:
            Output tensor of shape (batch_size, num_instances, output_dim).
        """
        for i in range(len(self.layers) - 1):
            x = torch.nn.functional.relu(self.layers[i](x))
            x = self.dropouts[i](x)
        x = self.layers[-1](x)

        return x
