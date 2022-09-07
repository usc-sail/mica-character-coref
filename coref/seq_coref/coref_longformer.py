"""Longformer model."""

from torch import nn
from transformers import LongformerTokenizer, LongformerModel


class CorefLongformerModel(nn.Module):
    """Coreference Resolution Model for English using the Longformer model.
    """

    def __init__(self, use_large: bool = False) -> None:
        super().__init__()

        model_size = "large" if use_large else "base"
        self.tokenizer: LongformerTokenizer = (
            LongformerTokenizer.from_pretrained(
                f"allenai/longformer-{model_size}-4096"))
        self.longformer: LongformerModel = LongformerModel.from_pretrained(
            f"allenai/longformer-{model_size}-4096")

        self.longformer_hidden_size: int = self.longformer.config.hidden_size
        self.n_labels = 3
        self.label_embedding_size = 10
        self.label_embedding = nn.Embedding(self.n_labels,
                                            self.label_embedding_size)
        self.gru_hidden_size = self.longformer_hidden_size
        self.gru = nn.GRU(
            self.longformer_hidden_size + self.label_embedding_size,
            self.gru_hidden_size, bidirectional=True)
        self.token_classifier = nn.Linear(self.gru_hidden_size, self.n_labels)
        