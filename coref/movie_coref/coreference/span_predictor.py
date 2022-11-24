import torch
from torch import nn

class SpanPredictor(nn.Module):
    """Span Predictor Model"""

    def __init__(self, word_embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.max_left = -1
        self.max_right = -1
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
        self.emb = nn.Embedding(128, 64)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
    
    def forward(
        self, 
        embeddings: torch.FloatTensor, 
        head_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Calculates span start/end scores of words for each head_id.

        Args:
            embeddings (torch.FloatTensor): word embeddings [n_words, emb_size]
            parse_ids (torch.LongTensor): word screenplay-parse ids [n_words]
            heads_ids (torch.LongTensor): word indices of heads of character
                mentions [n_heads]

        Returns:
            torch.FloatTensor: span start/end scores [n_heads, n_words, 2]
        """
        # Obtain distance embedding indices [n_heads, n_words]
        relative_positions = (head_ids.unsqueeze(1) - torch.arange(embeddings.shape[0], device=self.device).unsqueeze(0))
        emb_ids = relative_positions + 63
        emb_ids[(emb_ids < 0) + (emb_ids > 126)] = 127

        # Obtain valid positions boolean mask, [n_heads, n_words]
        valid_positions = (-self.max_right <= relative_positions) & (relative_positions <= self.max_left)

        # pair_matrix contains concatenated head word embedding + 
        # candidate word embedding + distance embedding, for each candidate 
        # among the words in valid positions for the head word
        # [total_n_candidates, emb_size x 2 + distance_emb_size]
        rows, cols = valid_positions.nonzero(as_tuple=True)
        pair_matrix = torch.cat((embeddings[head_ids[rows]], embeddings[cols], self.emb(emb_ids[rows, cols])), dim=1)

        # padding_mask: [n_heads, max_segment_len]
        lengths = valid_positions.sum(dim=1)
        padding_mask = torch.arange(0, lengths.max(), device=self.device).unsqueeze(0)
        padding_mask = (padding_mask < lengths.unsqueeze(1))

        # [n_heads, max_segment_len, emb_size x 2 + distance_emb_size]
        # This is necessary to allow the convolution layer to look at several
        # word scores
        padded_pairs = torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=self.device)
        padded_pairs[padding_mask] = pair_matrix

        # res: [n_heads, n_candidates, 2]
        res = self.ffnn(padded_pairs)
        res = self.conv(res.permute(0, 2, 1)).permute(0, 2, 1)

        # scores: [n_heads, n_words, 2]
        scores = torch.full((head_ids.shape[0], embeddings.shape[0], 2), float('-inf'), device=self.device)
        scores[rows, cols] = res[padding_mask]

        # Make sure that start <= head <= end during inference
        if not self.training:
            valid_starts = torch.log((relative_positions >= 0).to(torch.float))
            valid_ends = torch.log((relative_positions <= 0).to(torch.float))
            valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
            return scores + valid_positions

        return scores