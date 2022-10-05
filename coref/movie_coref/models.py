"""PyTorch modules for fine-tuning wl-RoBERTa on movie coreference data
"""

import torch
from torch import nn
from transformers import AutoModel

class CharacterRecognition(nn.Module):
    """Character Recognition Model.
    """

    def __init__(self, 
                 encoder_name: str,
                 num_parse_tags: int,
                 parse_tag_embedding_size: int,
                 gru_hidden_size: int,
                 gru_num_layers: int,
                 gru_dropout: float,
                 gru_bidirectional: bool,
                 num_labels: int) -> None:
        """Initializer for Character Recognition Model.

        Args:
            encoder_name: Language model encoder name from transformers hub
                e.g. bert-base-cased
            num_parse_tags: Parse tag set size
            parse_tag_embedding_size: Embedding size of the parse tags
            gru_hidden_size: Hidden size of the GRU
            gru_num_layers: Number of layers of the GRU
            gru_dropout: Dropout used between the GPU layers
            gru_bidirectional: If true, the GRU is bidirectional
            num_labels: Number of labels in the label set. 2 if label_type =
                "head" or 3 if label_type = "span"
        """
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.subtoken = nn.Linear(self.encoder_hidden_size, 1)
        self.parse_embedding = nn.Embedding(
            num_parse_tags, parse_tag_embedding_size)
        self.gru_input_size = (self.encoder_hidden_size +
                               parse_tag_embedding_size)
        self.gru_output_size = gru_hidden_size * (1 + int(gru_bidirectional))
        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size,
                          num_layers=gru_num_layers, batch_first=True,
                          dropout=gru_dropout, bidirectional=gru_bidirectional)
        self.output = nn.Linear(self.gru_output_size, num_labels)
    
    @property
    def device(self) -> torch.device:
        """A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules).
        TODO: sabyasachee accelerator
        """
        return next(self.parameters()).device
    
    def forward(self, subtoken_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_offset: torch.Tensor, parse_ids: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        """Forward propagation for the Character Recognition Model.

        Args:
            subtoken_ids: `batch_size x max_n_subtokens` Long Tensor
            attention_mask: `batch_size x max_n_subtokens` Float/Long Tensor
            token_offset: `batch_size x max_n_tokens x 2` Long Tensor
            parse_ids: `batch_size x max_n_tokens` Long Tensor
            labels: `batch_size x max_n_tokens` Long Tensor or None
        
        Returns:
            The loss value if labels are given, else the logits 
            `batch_size x max_n_tokens x num_labels` Float Tensor
        """
        batch_size = len(subtoken_ids)

        # subtoken_embedding = batch_size x max_n_subtokens x encoder_hidden_size
        encoder_output = self.encoder(subtoken_ids, attention_mask)
        subtoken_embedding = encoder_output.last_hidden_state

        # subtoken_attn = batch_size * max_n_tokens x batch_size * max_n_subtokens
        _subtoken_embedding = subtoken_embedding.view(-1, self.encoder_hidden_size)
        subtoken_attn = self._attn_scores(_subtoken_embedding,
                                          token_offset.view(-1, 2))
        
        # token_embedding = batch_size x max_n_tokens x encoder_hidden_size
        token_embedding = torch.mm(
            subtoken_attn, _subtoken_embedding).reshape(
                batch_size, -1, self.encoder_hidden_size)
        
        # gru_input = batch_size x max_n_tokens x (encoder_hidden_size +
        # parse_tag_embedding_size)
        parse_input = self.parse_embedding(parse_ids)
        gru_input = torch.cat((token_embedding, parse_input), dim=0).contiguous()

        # logits = batch_size x max_n_tokens x num_labels
        gru_output, _ = self.gru(gru_input)
        logits = self.output(gru_output)

        if labels is None:
            return logits
        else:
            token_attention_mask = torch.any(subtoken_attn > 0, dim=1).reshape(
                batch_size, -1)
            loss = compute_loss(logits, labels, token_attention_mask,
                                self.num_labels)
            return loss

    def _attn_scores(self,
                     subtoken_embeddings: torch.FloatTensor,
                     token_offset: torch.LongTensor) -> torch.FloatTensor:
        """ Calculates attention scores for each of the subtokens of a token.

        Args:
            subtoken_embedding: `n_subtokens x embedding_size` Float Tensor,
                embeddings for each subtoken
            token_offset: `n_tokens x 2` Long Tensor, subtoken offset of each
                token

        Returns:
            torch.FloatTensor: `n_tokens x n_subtokens` Float Tensor, attention
            weights for each subtoken of a token
        """
        n_subtokens, n_tokens = len(subtoken_embeddings), len(token_offset)
        token_begin, token_end = token_offset[:,0], token_offset[:,1]
        
        # attn_mask: n_tokens x n_subtokens, contains -∞ for subtokens outside
        # the token's offsets and 0 for subtokens inside the token's offsets
        attn_mask = torch.arange(0, n_subtokens, device=self.device).expand(
            (n_tokens, n_subtokens))
        attn_mask = ((attn_mask >= token_begin.unsqueeze(1)) * 
                     (attn_mask <= token_end.unsqueeze(1)))
        attn_mask = torch.log(attn_mask.to(torch.float))

        # attn_scores: 1 x n_subtokens
        attn_scores = self.subtoken(subtoken_embeddings).T

        # attn_scores: n_tokens x n_subtokens
        attn_scores = attn_scores.expand((n_tokens, n_subtokens))

        # -∞ for subtokens outside the token's offsets and attn_scores for
        # inside the token's offsets
        attn_scores = attn_mask + attn_scores
        del attn_mask

        # subtoken_attn contains 0 for subtokens outside the token's offsets
        subtoken_attn = torch.softmax(attn_scores, dim=1)
        return subtoken_attn
    
def compute_loss(
    logits: torch.FloatTensor, label_ids: torch.LongTensor,
    attn_mask: torch.FloatTensor, n_labels: int) -> torch.FloatTensor:
    """Compute cross entropy loss"""
    active_labels = label_ids[attn_mask == 1.]
    active_logits = logits.flatten(0, 1)[attn_mask.flatten() == 1.]
    label_distribution = torch.bincount(active_labels,
        minlength=n_labels)
    class_weight = torch.sqrt(len(active_labels)/(1 + label_distribution))
    cross_entrop_loss_fn = nn.CrossEntropyLoss(weight=class_weight, 
        reduction="mean")
    loss = cross_entrop_loss_fn(active_logits, active_labels)
    return loss