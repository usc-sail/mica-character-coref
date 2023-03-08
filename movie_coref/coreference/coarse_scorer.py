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
    
    def forward(self, word_embeddings: torch.Tensor, character_scores: torch.Tensor = None,
                score_succeeding: bool = False, prune: bool = True) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward propagation of coarse scoring module

        Args:
            word_embeddings (Tensor[n, d]): word embeddings
            character_scores (Tensor[n]): word-wise character scores. If None, then it is not added to final score
            score_succeeding (bool): score future words, otherwise put -inf
            prune (bool): retain topk scoring words
        
        Returns:
            if prune
                top_scores (Tensor[n, k]): top scores
                indices (Tensor[n, k]): indices of the top-scoring words
            else
                scores (Tensor[n, n]): scores
        """
        # [n_words, n_words]
        if score_succeeding:
            pair_mask = 0
        else:
            pair_mask = torch.arange(word_embeddings.shape[0])
            pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
            pair_mask = torch.log((pair_mask > 0).to(torch.float))
            pair_mask = pair_mask.to(self.device)
        bilinear_scores = self.dropout(self.bilinear(word_embeddings)).mm(word_embeddings.T)
        if character_scores is not None:
            rough_scores = (pair_mask + bilinear_scores + character_scores.unsqueeze(dim=1)
                            + character_scores.unsqueeze(dim=0))
        else:
            rough_scores = pair_mask + bilinear_scores
        if prune:
            top_scores, indices = torch.topk(rough_scores, k=min(self.topk, len(rough_scores)), dim=1, sorted=False)
            return top_scores, indices
        else:
            return rough_scores