"""Longformer model."""

import numpy as np
import torch
from torch import nn
from transformers import LongformerTokenizer, LongformerModel
from transformers.models.longformer import modeling_longformer

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
            self.gru_hidden_size, batch_first=True, bidirectional=True)
        self.token_classifier = nn.Linear(2 * self.gru_hidden_size, 
                                          self.n_labels)
    
    def forward(self, 
                token_ids: torch.LongTensor, 
                mention_ids: torch.IntTensor, 
                attn_mask: torch.FloatTensor, 
                global_attn_mask: torch.FloatTensor, 
                label_ids: torch.IntTensor | None = None) -> (
                tuple[torch.IntTensor, torch.FloatTensor] | torch.IntTensor):
        """Forward propagation. If label_ids is None, return predictions only.
        Otherwise, return both predictions and loss, which is used for
        backpropagation.
        """
        device = next(self.parameters()).device
        longformer_output: (
            modeling_longformer.LongformerBaseModelOutputWithPooling) = (
                self.longformer(token_ids, attn_mask, global_attn_mask))
        longformer_hidden: torch.FloatTensor = (
            longformer_output.last_hidden_state)
        mention_embedding: torch.FloatTensor = self.label_embedding(mention_ids)
        gru_input: torch.FloatTensor = torch.cat(
            (longformer_hidden, mention_embedding), dim=2)
        gru_output: torch.FloatTensor = self.gru(gru_input)[0]
        logits: torch.FloatTensor = self.token_classifier(gru_output).int()
        predictions: torch.IntTensor = logits.argmax(dim=2)

        if label_ids is not None:
            active_labels = label_ids[attn_mask == 1.]
            active_logits = logits.flatten(0, 1)[attn_mask.flatten() == 1.]
            label_distribution = np.bincount(active_labels, 
                                             minlength=self.n_labels)
            class_weight = torch.FloatTensor(1/(1 + label_distribution)).to(
                device)
            cross_entrop_loss_fn = nn.CrossEntropyLoss(weight=class_weight, 
                                                       reduction="mean")
            loss = cross_entrop_loss_fn(active_logits, active_labels)
            return predictions, loss
        else:
            return predictions 