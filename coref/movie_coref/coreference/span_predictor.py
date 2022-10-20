import torch
from torch import nn

class SpanPredictor(nn.Module):

    def __init__(self, word_embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(word_embedding_size * 2 + 64, word_embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(word_embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(64, 4, 3, 1, 1),
            nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.distance_embedding = nn.Embedding(128, 64)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self._device = device