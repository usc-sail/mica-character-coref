import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, word_embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.Linear(word_embedding_size, 1)
        self.dropout = nn.Dropout(dropout)
        self._device = torch.device("cpu")
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
    
    def forward(self, 
                subword_embeddings: torch.Tensor,
                word_to_subword_offset: torch.Tensor) -> torch.Tensor:
        # subword_embeddings: [n_subwords, embedding_size]
        # word_to_subword_offset: [n_words, 2]
        n_subwords = len(subword_embeddings)
        n_words = len(word_to_subword_offset)

        # attn_mask: [n_words, n_subwords]
        # with 0 at positions belonging to the words and -inf elsewhere
        attn_mask = torch.arange(
            0, n_subwords, device=self.device).expand((n_words, n_subwords))
        attn_mask = ((attn_mask >= word_to_subword_offset[:,0].unsqueeze(1)) *
                    (attn_mask < word_to_subword_offset[:,1].unsqueeze(1)))
        attn_mask = torch.log(attn_mask.to(torch.float))

        # attn_scores: [n_words, n_subwords]
        # with attention weights at positions belonging to the words and 0 
        # elsewhere
        attn_scores = self.attn(subword_embeddings).T
        attn_scores = attn_scores.expand((n_words, n_subwords))
        attn_scores = attn_mask + attn_scores
        del attn_mask
        attn_scores = torch.softmax(attn_scores, dim=1)

        # word_embeddings: [n_words, embedding_size]
        word_embeddings = attn_scores.mm(subword_embeddings)
        word_embeddings = self.dropout(word_embeddings)
        return word_embeddings