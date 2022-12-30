import torch
from torch import nn

class FineScorer(nn.Module):

    def __init__(self, word_embedding_size: int, dropout: float) -> None:
        super().__init__()
        pairwise_encoding_size = 3*word_embedding_size + 3*20
        self.hidden = nn.Sequential(
            nn.Linear(pairwise_encoding_size, word_embedding_size), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout))
        self.out = torch.nn.Linear(word_embedding_size, 1)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def _get_pair_matrix(
        self, 
        all_embeddings: torch.Tensor,
        embeddings: torch.Tensor,
        features: torch.Tensor,
        indices: torch.Tensor) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_embeddings: [all_words, embedding_size]
            embeddings: [n_words, embedding_size]
            features: [n_words, n_ants, 3*20]
            indices: [n_words, n_ants]
        Returns:
            pair_matrix: [n_words, n_ants, 3*word_embedding_size + 3*20]
        """
        emb_size = embeddings.shape[1]
        n_ants = features.shape[1]
        a_mentions = embeddings.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_embeddings[indices]
        similarity = a_mentions * b_mentions
        out = torch.cat((a_mentions, b_mentions, similarity, features), dim=2)
        return out

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(self.hidden(x))
        return x.squeeze(dim=2)

    def _add_dummy(self, tensor: torch.Tensor):
        shape = list(tensor.shape)
        shape[1] = 1
        dummy = torch.full(
            shape, 1e-7, device=self.device, dtype=tensor.dtype)
        return torch.cat((dummy, tensor), dim=1)

    def forward(
        self,
        all_embeddings: torch.Tensor = None,
        embeddings: torch.Tensor = None,
        features: torch.Tensor = None,
        indices: torch.Tensor = None,
        scores: torch.Tensor = None,
        pair_matrix: torch.Tensor = None) -> torch.Tensor:
        if pair_matrix is not None:
            return self._ffnn(pair_matrix)
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(
            all_embeddings, embeddings, features, indices)

        # [batch_size, 1 + n_ants]
        scores = scores + self._ffnn(pair_matrix)
        scores = self._add_dummy(scores)

        return scores