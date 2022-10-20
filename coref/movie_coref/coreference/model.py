"""Python class including PyTorch modules that find character coreference chains
in movie screenplay document.
"""
from mica_text_coref.coref.movie_coref.coreference.encoder import Encoder
from mica_text_coref.coref.movie_coref.coreference.character_recognizer import CharacterRecognizer
from mica_text_coref.coref.movie_coref.coreference.coarse_scorer import CoarseScorer
from mica_text_coref.coref.movie_coref.coreference.fine_scorer import FineScorer
from mica_text_coref.coref.movie_coref.coreference.span_predictor import SpanPredictor

import itertools
import torch
from transformers import RobertaTokenizerFast

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
        dropout: float) -> None:
        self.tokenizer: RobertaTokenizerFast = (
            RobertaTokenizerFast.from_pretrained(
                "roberta-large", use_fast=True, add_prefix_space=True))
        self.tokenizer_map = {
            ".": ["."], ",": [","], "!": ["!"], "?": ["?"],":":[":"], 
            ";":[";"], "'s": ["'s"]}
        self.encoder = Encoder(dropout)
        word_embedding_size = self.encoder.word_embedding_size
        self.character_recognizer = CharacterRecognizer(
            word_embedding_size, tag_embedding_size, parsetag_size, postag_size,
            nertag_size, gru_nlayers, gru_hidden_size, gru_bidirectional,
            dropout)
        self.coarse_scorer = CoarseScorer(word_embedding_size, topk, dropout)
        self.fine_scorer = FineScorer(word_embedding_size, dropout)
        self.span_predictor = SpanPredictor(word_embedding_size, dropout)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:    
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.encoder.device = device
        self.character_recognizer.device = device
        self.coarse_scorer.device = device
        self.fine_scorer.device = device
        self.span_predictor.device = device
    
    def _rename_keys(
        self, 
        weights_dict: dict[str, torch.Tensor], 
        renaming_schema: dict[str, str]) -> dict[str, torch.Tensor]:
        for old_key, new_key in renaming_schema.items():
            weights_dict[new_key] = weights_dict.pop(old_key)
        return weights_dict

    def load_weights(self, weights_path: str):
        weights = torch.load(weights_path, map_location="cpu")
        encoder_weights = weights["bert"]
        attn_weights = self._rename_keys(
            weights["we"], {"attn.weight": "weight", "attn.bias": "bias"})
        coarse_scorer_weights = self._rename_keys(
            weights["rough_scorer"], 
            {"bilinear.weight": "scorer.weight",
            "bilinear.bias": "scorer.bias"})
        fine_scorer_weights = {**weights["pw"], **weights["a_scorer"]}
        fine_scorer_weights = self._rename_keys(
            fine_scorer_weights, 
            {"genre_emb.weight": "genre_embedding.weight",
            "distance_emb.weight": "distance_embedding.weight",
            "speaker_emb.weight": "speaker_embedding.weight",
            "hidden.0.weight": "scorer.0.weight",
            "hidden.0.bias": "scorer.0.bias",
            "out.weight": "scorer.3.weight",
            "out.bias": "scorer.3.bias"})
        sp_weights = self._rename_keys(
            weights["sp"], {"emb.weight": "distance_embedding.weight"})
        self.encoder.encoder.load_state_dict(encoder_weights, strict=False)
        self.encoder.attn.load_state_dict(attn_weights)
        self.coarse_scorer.load_state_dict(coarse_scorer_weights)
        self.fine_scorer.load_state_dict(fine_scorer_weights)
        self.span_predictor.load_state_dict(sp_weights)
    
    def bert_parameters(self):
        return self.encoder.encoder.parameters()
    
    def parameters(self):
        return itertools.chain(
            self.encoder.attn.parameters(), 
            self.character_recognizer.parameters(),
            self.coarse_scorer.parameters(),
            self.fine_scorer.parameters(),
            self.span_predictor.parameters())
    
    def modules(self):
        return [
            self.encoder, self.character_recognizer, self.coarse_scorer, 
            self.fine_scorer, self.span_predictor]
    
    def train(self):
        for module in self.modules():
            module.train()
    
    def eval(self):
        for module in self.modules:
            module.eval()