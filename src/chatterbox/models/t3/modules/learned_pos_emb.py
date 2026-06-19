from typing import Union

import torch
from torch import nn, Tensor


class LearnedPositionEmbeddings(nn.Module):
    """Class representing learned positional embeddings for a sequence.
    Initializes an embedding layer that learns to represent positions in a sequence.
    Provides a forward method to retrieve positional embeddings based on input indices.
    """
    def __init__(self, seq_len, model_dim, init=.02):
        """Initializes positional embeddings for a sequence.
        Args:
        seq_len (int): Length of the sequence.
        model_dim (int): Dimensionality of the model embeddings.
        init (float, optional): Standard deviation for initializing weights. Default is .02.
        Returns:
        nn.Embedding: Positional embedding layer.
        Gets fixed positional embeddings at a given index or set of indices.
        Args:
        idx (Union[int, Tensor]): Index or tensor of indices to retrieve embeddings for.
        Returns:
        Tensor: Embeddings corresponding to the provided indices.
        """
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        """
        Returns positional embeddings for index 0 up to the length of x
        """
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, idx: 'Union[int, Tensor]'):
        """
        Args:
            idx: scalar int or an integer tensor of shape (T,) or (B, T)
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        """
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx = torch.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx)  # (B, T, dim)
