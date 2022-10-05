import torch
import torch.nn as nn


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class BOWEmbedding(nn.Module):
    def __init__(self, max_value, n_channels, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_channels * max_value, embedding_dim)
        self.n_channels = n_channels
        self.apply(initialize_parameters)

    def forward(self, inputs):
        flat_inputs = inputs.flatten(0, -2)

        offsets = torch.Tensor([i * self.max_value for i in range(self.n_channels)]).to(
            inputs.device
        )
        offsetted = (flat_inputs + offsets[None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.flatten(-2, -1)

        return each_embedding_flat.unflatten(0, inputs.shape[:-1])
