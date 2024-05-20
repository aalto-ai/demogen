import torch
import torch.nn as nn
import numpy as np


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class BOWEmbedding(nn.Module):
    def __init__(self, max_value, n_channels, embedding_dim, state_component_lengths=None):
        super().__init__()
        state_component_lengths = (
            state_component_lengths if state_component_lengths is not None
            else [max_value] * n_channels
        )

        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(sum(state_component_lengths), embedding_dim)
        self.n_channels = n_channels
        self.register_buffer(
            "state_component_offsets",
            torch.from_numpy(
                np.cumsum([0] + (
                    state_component_lengths
                )[:-1])
            ),
            persistent=False
        )
        self.apply(initialize_parameters)

    def forward(self, inputs):
        flat_inputs = inputs.flatten(0, -2)
        offsets = self.state_component_offsets
        offsetted = (flat_inputs + offsets[None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.flatten(-2, -1)

        return each_embedding_flat.unflatten(0, inputs.shape[:-1])
