"""PyTorch module for modeling the probability of a word being the head of a
character mention.
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
                 num_pos_tags: int,
                 pos_tag_embedding_size: int,
                 num_ner_tags: int,
                 ner_tag_embedding_size: int,
                 gru_hidden_size: int,
                 gru_num_layers: int,
                 gru_dropout: float,
                 gru_bidirectional: bool,
                 num_labels: int,
                 use_parse_tags: bool = False,
                 use_pos_tags: bool = False,
                 use_ner_tags: bool = False,
                 use_is_pronoun: bool = False,
                 use_is_punctuation: bool = False,
                 gradient_checkpointing: bool = False,
                 class_weights: list[float] = None) -> None:
        """Initializer for Character Recognition Model.

        Args:
            encoder_name: Language model encoder name from transformers hub
                e.g. bert-base-cased
            num_parse_tags: Parse tag set size
            parse_tag_embedding_size: Embedding size of the parse tags
            num_pos_tags: Pos tag set size
            pos_tag_embedding_size: Embedding size of the pos tags
            num_ner_tags: Ner tag set size
            ner_tag_embedding_size: Embedding size of the ner tags
            gru_hidden_size: Hidden size of the GRU
            gru_num_layers: Number of layers of the GRU
            gru_dropout: Dropout used between the GPU layers
            gru_bidirectional: If true, the GRU is bidirectional
            num_labels: Number of labels in the label set. 2 if label_type =
                "head" or 3 if label_type = "span"
            use_parse_tags: If true, use parse tags.
            use_pos_tags: If true, use pos tags.
            use_ner_tags: If true, use ner tags.
            gradient_checkpointing: If true, enable gradient checkpointing in
                encoder
            class_weights: List of class weights to use in cross entropy loss.
                If none, class weight is batch_samples/class_samples
        """
        super().__init__()
        self.num_parse_tags = num_parse_tags
        self.num_pos_tags = num_pos_tags
        self.num_ner_tags = num_ner_tags
        self.num_labels = num_labels
        self.use_parse_tags = use_parse_tags
        self.use_pos_tags = use_pos_tags
        self.use_ner_tags = use_ner_tags
        self.use_is_pronoun = use_is_pronoun
        self.use_is_punctuation = use_is_punctuation
        self.encoder = AutoModel.from_pretrained(
            encoder_name, add_pooling_layer=False)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.subtoken = nn.Linear(self.encoder_hidden_size, 1)
        self.gru_input_size = self.encoder_hidden_size
        if self.use_parse_tags:
            self.parse_embedding = nn.Embedding(num_parse_tags,
                                                parse_tag_embedding_size)
            self.gru_input_size += parse_tag_embedding_size
        if self.use_pos_tags:
            self.pos_embedding = nn.Embedding(num_pos_tags,
                                              pos_tag_embedding_size)
            self.gru_input_size += pos_tag_embedding_size
        if self.use_ner_tags:
            self.ner_embedding = nn.Embedding(num_ner_tags,
                                              ner_tag_embedding_size)
            self.gru_input_size += ner_tag_embedding_size
        if self.use_is_pronoun:
            self.gru_input_size += 1
        if self.use_is_punctuation:
            self.gru_input_size += 1
        self.gru_output_size = gru_hidden_size * (1 + int(gru_bidirectional))
        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size,
                          num_layers=gru_num_layers, batch_first=True,
                          dropout=gru_dropout, bidirectional=gru_bidirectional)
        self.output = nn.Linear(self.gru_output_size, num_labels)
        self._device = torch.device("cpu")
        self._class_weights = class_weights
    
    @property
    def device(self) -> torch.device:
        """Getter for model device."""
        return self._device
    
    @device.setter
    def device(self, device):
        """Setter for model device. Used by accelerate."""
        self._device = device
    
    def forward(self, subtoken_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_offset: torch.Tensor, parse_ids: torch.Tensor,
                pos_ids: torch.Tensor, ner_ids: torch.Tensor,
                is_pronoun: torch.Tensor, is_punctuation: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Forward propagation for the Character Recognition Model.

        Args:
            subtoken_ids: `batch_size x max_n_subtokens` Long Tensor
            attention_mask: `batch_size x max_n_subtokens` Float/Long Tensor
            token_offset: `batch_size x max_n_tokens x 2` Long Tensor
            parse_ids: `batch_size x max_n_tokens` Long Tensor
            pos_ids: `batch_size x max_n_tokens` Long Tensor
            ner_ids: `batch_size x max_n_tokens` Long Tensor
            is_pronoun: `batch_size x max_n_tokens` Float Tensor
            is_punctuation: `batch_size x max_n_tokens` Float Tensor
            labels: `batch_size x max_n_tokens` Long Tensor
        
        Returns:
            Return the loss value if model is begin trained, else the logits 
            `batch_size x max_n_tokens x num_labels` Float Tensor
        """
        batch_size = len(subtoken_ids)

        # subtoken_embedding = batch_size x max_n_subtokens x 
        #                      encoder_hidden_size
        encoder_output = self.encoder(subtoken_ids, attention_mask)
        subtoken_embedding = encoder_output.last_hidden_state

        # subtoken_attn = batch_size * max_n_tokens x batch_size * 
        #                 max_n_subtokens
        _subtoken_embedding = subtoken_embedding.view(
            -1, self.encoder_hidden_size)
        subtoken_attn = self._attn_scores(_subtoken_embedding,
                                          token_offset.view(-1, 2))
        
        # token_embedding = batch_size x max_n_tokens x encoder_hidden_size
        token_embedding = torch.mm(
            subtoken_attn, _subtoken_embedding).reshape(
                batch_size, -1, self.encoder_hidden_size)
        
        # gru_input = batch_size x max_n_tokens x (encoder_hidden_size +
        # parse_tag_embedding_size(opt.) + pos_tag_embedding_size(opt.) + 
        # ner_tag_embedding_size(opt.))
        gru_input_tensors = [token_embedding]
        if self.use_parse_tags:
            parse_input = self.parse_embedding(parse_ids)
            gru_input_tensors.append(parse_input)
        if self.use_pos_tags:
            pos_input = self.pos_embedding(pos_ids)
            gru_input_tensors.append(pos_input)
        if self.use_ner_tags:
            ner_input = self.ner_embedding(ner_ids)
            gru_input_tensors.append(ner_input)
        if self.use_is_pronoun:
            gru_input_tensors.append(is_pronoun.view(batch_size, -1, 1))
        if self.use_is_punctuation:
            gru_input_tensors.append(is_punctuation.view(batch_size, -1, 1))
        gru_input = torch.cat(gru_input_tensors, dim=2).contiguous()

        # logits = batch_size x max_n_tokens x num_labels
        gru_output, _ = self.gru(gru_input)
        logits = self.output(gru_output)

        # token_attention_mask = batch_size x max_n_tokens
        token_attention_mask = torch.any(subtoken_attn > 0, dim=1).reshape(
            batch_size, -1)
        loss = compute_loss(logits, labels, token_attention_mask,
                            self.num_labels, self._class_weights,
                            self.device)
        return loss, logits

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
    attn_mask: torch.FloatTensor, n_labels: int,
    class_weights: list[float] = None,
    device: torch.device = torch.device("cpu")) -> torch.FloatTensor:
    """Compute cross entropy loss"""
    active_labels = label_ids[attn_mask == 1.]
    active_logits = logits.flatten(0, 1)[attn_mask.flatten() == 1.]
    if class_weights is None:
        label_distribution = torch.bincount(active_labels, minlength=n_labels)
        class_weight = len(active_labels)/(1 + label_distribution)
    else:
        assert len(class_weights) == n_labels, (
            "Length of the list of class weights should equal number of labels")
        class_weight = torch.FloatTensor(class_weights).to(device)
    cross_entrop_loss_fn = nn.CrossEntropyLoss(weight=class_weight, 
        reduction="mean")
    loss = cross_entrop_loss_fn(active_logits, active_labels)
    return loss