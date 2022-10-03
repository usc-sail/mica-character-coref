"""PyTorch modules for fine-tuning wl-RoBERTa on movie coreference data
"""

import torch
from torch import nn
from transformers import AutoModel

class CharacterRecognition(nn.Module):
    """Character Recognition Model.
    """

    def __init__(self, encoder_name: str, num_labels: int, gru_hidden_size: int,
                 gru_num_layers: int, dropout: float,
                 gru_bidirectional = False) -> None:
        """Initializer for Character Recognition Model.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        self.subtoken = nn.Linear(encoder_hidden_size, 1)
        self.gru = nn.GRU(encoder_hidden_size, gru_hidden_size,
                          num_layers=gru_num_layers, batch_first=True,
                          dropout=dropout, bidirectional=gru_bidirectional)
        gru_output_size = gru_hidden_size * (1 + int(gru_bidirectional))
        self.output = nn.Linear(gru_output_size, num_labels)
    
    def forward(self, subtoken_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_offset: torch.Tensor, parse_ids: torch.Tensor,
                labels: torch.Tensor):