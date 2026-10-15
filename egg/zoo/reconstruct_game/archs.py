# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class Sender(nn.Module):
    """Encodes Sender's input into the initial hidden state of its RNN.

    input_layer="linear": one linear layer (as in egg.zoo.compo_vs_generalization)
    input_layer="mlp":    linear -> ReLU -> linear
    """

    def __init__(self, n_inputs: int, n_hidden: int, input_layer: str = "linear"):
        super().__init__()
        if input_layer == "linear":
            self.fc1 = nn.Linear(n_inputs, n_hidden)
        elif input_layer == "mlp":
            self.fc1 = nn.Sequential(
                nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden)
            )
        else:
            raise ValueError(f"Unknown input layer: {input_layer}")

    def forward(self, x, _aux_input=None):
        return self.fc1(x)


class Receiver(nn.Module):
    """Maps the final hidden state of Receiver's RNN to the concatenated per-attribute logits."""

    def __init__(self, n_hidden: int, n_outputs: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input=None, _aux_input=None):
        return self.fc(self.dropout(x))
