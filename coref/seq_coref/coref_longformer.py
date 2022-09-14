"""Longformer model."""

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
                global_attn_mask: torch.FloatTensor) -> torch.FloatTensor:
        """Forward propagation"""
        longformer_output: (
            modeling_longformer.LongformerBaseModelOutputWithPooling) = (
                self.longformer(token_ids, attn_mask, global_attn_mask))
        longformer_hidden: torch.FloatTensor = (
            longformer_output.last_hidden_state)
        mention_embedding: torch.FloatTensor = self.label_embedding(mention_ids)
        gru_input: torch.FloatTensor = torch.cat(
            (longformer_hidden, mention_embedding), dim=2).contiguous()
        self.gru.flatten_parameters()
        gru_output: torch.FloatTensor = self.gru(gru_input)[0]
        logits: torch.FloatTensor = self.token_classifier(gru_output)
        return logits

def compute_loss(logits: torch.FloatTensor, label_ids: torch.LongTensor,
    attn_mask: torch.FloatTensor, n_labels: int) -> torch.FloatTensor:
    """Compute cross entropy loss"""
    active_labels = label_ids[attn_mask == 1.]
    active_logits = logits.flatten(0, 1)[attn_mask.flatten() == 1.]
    label_distribution = torch.bincount(active_labels,
        minlength=n_labels)
    class_weight = 1/(1 + label_distribution)
    cross_entrop_loss_fn = nn.CrossEntropyLoss(weight=class_weight, 
        reduction="mean")
    loss = cross_entrop_loss_fn(active_logits, active_labels)
    return loss