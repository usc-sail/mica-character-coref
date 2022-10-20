import torch
from torch import nn

class FineScorer(nn.Module):

    def __init__(self, word_embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.genre_embedding = nn.Embedding(7, 20)
        self.distance_embedding = nn.Embedding(9, 20)
        self.speaker_embedding = nn.Embedding(2, 20)
        pairwise_encoding_size = 3*word_embedding_size + 3*20
        self.scorer = nn.Sequential(
            nn.Linear(pairwise_encoding_size, word_embedding_size), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout),
            nn.Linear(word_embedding_size, 1))
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self._device = device