import torch
from torch import nn

class CharacterRecognizer(nn.Module):

    def __init__(
        self,
        word_embedding_size: int,
        tag_embedding_size: int,
        parsetag_size: int,
        postag_size: int,
        nertag_size: int,
        gru_nlayers: int,
        gru_hidden_size: int,
        gru_bidirectional: bool,
        dropout: float) -> None:
        super().__init__()
        gru_input_size = word_embedding_size + 3*tag_embedding_size + 2
        gru_output_size = gru_hidden_size * (1 + gru_bidirectional)
        self.parsetag_embedding = nn.Embedding(parsetag_size, tag_embedding_size)
        self.postag_embedding = nn.Embedding(postag_size, tag_embedding_size)
        self.nertag_embedding = nn.Embedding(nertag_size, tag_embedding_size)
        self.gru = nn.GRU(gru_input_size, gru_hidden_size, num_layers=gru_nlayers, batch_first=True, bidirectional=gru_bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(gru_output_size, 1)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
    
    def forward(self, word_embeddings: torch.Tensor, parsetag_ids: torch.Tensor, postag_ids: torch.Tensor, nertag_ids: torch.Tensor, is_pronoun: torch.Tensor, is_punctuation: torch.Tensor
        ) -> torch.Tensor:
        # gru_input: [n_sequences, n_sequence_words, 
        # embedding_size + 3 x tag_embedding_size + 2]
        parsetag_embeddings = self.parsetag_embedding(parsetag_ids)
        postag_embeddings = self.postag_embedding(postag_ids)
        nertag_embeddings = self.nertag_embedding(nertag_ids)
        gru_input = torch.cat(
            [word_embeddings, 
            parsetag_embeddings, 
            postag_embeddings, 
            nertag_embeddings,
            is_pronoun.unsqueeze(dim=2),
            is_punctuation.unsqueeze(dim=2)
            ], dim=2).contiguous()
        
        # character_scores: [n_words]
        gru_output, _ = self.gru(gru_input)
        gru_output = self.dropout(gru_output)
        character_scores = self.output(gru_output).squeeze(dim=2)
        return character_scores