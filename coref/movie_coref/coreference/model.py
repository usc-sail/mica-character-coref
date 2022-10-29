"""Python class including PyTorch modules that find character coreference chains
in movie screenplay document.
"""
from mica_text_coref.coref.movie_coref.coreference.encoder import Encoder
from mica_text_coref.coref.movie_coref.coreference.character_recognizer import CharacterRecognizer
from mica_text_coref.coref.movie_coref.coreference.pairwise_encoder import PairwiseEncoder
from mica_text_coref.coref.movie_coref.coreference.coarse_scorer import CoarseScorer
from mica_text_coref.coref.movie_coref.coreference.fine_scorer import FineScorer
from mica_text_coref.coref.movie_coref.coreference.span_predictor import SpanPredictor

import itertools
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import RobertaTokenizerFast, RobertaModel

class MovieCoreference:

    def __init__(
        self,
        parsetag_size: int,
        postag_size: int,
        nertag_size: int,
        tag_embedding_size: int,
        gru_nlayers: int,
        gru_hidden_size: int,
        gru_bidirectional: bool,
        topk: int,
        bce_weight: float,
        dropout: float) -> None:
        self.tokenizer: RobertaTokenizerFast = (
            RobertaTokenizerFast.from_pretrained(
                "roberta-large", use_fast=True, add_prefix_space=True))
        self.tokenizer_map = {
            ".": ["."], ",": [","], "!": ["!"], "?": ["?"],":":[":"], 
            ";":[";"], "'s": ["'s"]}
        self.bert: RobertaModel = RobertaModel.from_pretrained(
            "roberta-large", add_pooling_layer=False)
        word_embedding_size = self.bert.config.hidden_size
        self.encoder = Encoder(word_embedding_size, dropout)
        self.character_recognizer = CharacterRecognizer(
            word_embedding_size, tag_embedding_size, parsetag_size, postag_size,
            nertag_size, gru_nlayers, gru_hidden_size, gru_bidirectional,
            dropout)
        self.pairwise_encoder = PairwiseEncoder(dropout)
        self.coarse_scorer = CoarseScorer(word_embedding_size, topk, dropout)
        self.fine_scorer = FineScorer(word_embedding_size, dropout)
        self.span_predictor = SpanPredictor(word_embedding_size, dropout)
        self.bce_weight = bce_weight
        self._device = torch.device("cpu")
        self._training = False

    @property
    def device(self) -> torch.device:    
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.encoder.device = device
        self.character_recognizer.device = device
        self.pairwise_encoder.device = device
        self.coarse_scorer.device = device
        self.fine_scorer.device = device
        self.span_predictor.device = device
    
    @property
    def training(self) -> bool:
        return self._training
    
    def _rename_keys(
        self, 
        weights_dict: dict[str, torch.Tensor], 
        renaming_schema: dict[str, str]) -> dict[str, torch.Tensor]:
        for old_key, new_key in renaming_schema.items():
            weights_dict[new_key] = weights_dict.pop(old_key)
        return weights_dict

    def load_weights(self, weights_path: str):
        weights = torch.load(weights_path, map_location="cpu")
        self.bert.load_state_dict(weights["bert"], strict=False)
        self.encoder.load_state_dict(weights["we"])
        self.coarse_scorer.load_state_dict(weights["rough_scorer"])
        self.pairwise_encoder.load_state_dict(weights["pw"])
        self.fine_scorer.load_state_dict(weights["a_scorer"])
        self.span_predictor.load_state_dict(weights["sp"])
    
    def bert_parameters(self):
        return self.bert.parameters()
    
    def parameters(self):
        return itertools.chain(
            self.encoder.parameters(), 
            self.character_recognizer.parameters(),
            self.coarse_scorer.parameters(),
            self.pairwise_encoder.parameters(),
            self.fine_scorer.parameters(),
            self.span_predictor.parameters())
    
    def modules(self):
        return [
            self.bert, self.encoder, self.character_recognizer, 
            self.coarse_scorer, self.pairwise_encoder, self.fine_scorer, 
            self.span_predictor]
    
    def train(self):
        for module in self.modules():
            module.train()
        self._training = True
    
    def eval(self):
        for module in self.modules:
            module.eval()
        self._training = False

    def coref_loss(self, scores: torch.Tensor, labels: torch.Tensor):
        bce_loss_fn = BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(torch.clamp(scores, min=-50, max=50), labels)
        gold = torch.logsumexp(scores + torch.log(labels), dim=1)
        total = torch.logsumexp(scores, dim=1)
        nlml_loss = (total - gold).mean()
        return nlml_loss + self.bce_weight * bce_loss
    
    def cr_loss(self, scores: torch.Tensor, labels: torch.Tensor):
        pos_weight = torch.Tensor(
            [(labels == 0).sum()/(1 + (labels == 1).sum())]).to(labels.device)
        bce_loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        loss = bce_loss_fn(scores, labels)
        return loss
    
    def sp_loss(
        self, scores: torch.FloatTensor, starts: torch.LongTensor, 
        ends: torch.LongTensor, avg_n_heads: int) -> torch.Tensor:
        """Span Prediction Loss.

        Args:
            scores: [n_heads, n_words, 2] Float Tensor
            starts: [n_heads] Long Tensor
            ends: [n_heads] Long Tensor
            avg_n_heads: Average number of head words per document
        
        Returns:
            span prediction loss: Tensor
        """
        loss_fn = CrossEntropyLoss(reduction="sum")
        loss = (
            loss_fn(scores[:, :, 0], starts) + 
            loss_fn(scores[:, :, 1], ends)) / avg_n_heads / 2
        return loss