import torch
from torch import nn

class CoarseScorer(nn.Module):

    def __init__(self,word_embedding_size: int, topk: int, dropout: float) -> None:
        super().__init__()
        self.topk = topk
        self.bilinear = nn.Linear(word_embedding_size, word_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
    
    def forward(self, word_embeddings: torch.Tensor, character_scores: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # [n_words, n_words]
        pair_mask = torch.arange(word_embeddings.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(self.device)
        bilinear_scores = self.dropout(self.bilinear(word_embeddings)).mm(word_embeddings.T)
        if character_scores is not None:
            rough_scores = (pair_mask + bilinear_scores + character_scores.unsqueeze(dim=1) + character_scores.unsqueeze(dim=0))
        else:
            rough_scores = pair_mask + bilinear_scores
        top_scores, indices = torch.topk(rough_scores, k=min(self.topk, len(rough_scores)), dim=1, sorted=False)
        return top_scores, indices