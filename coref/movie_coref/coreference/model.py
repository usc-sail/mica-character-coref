"""PyTorch module to find character mention heads, the pairwise coreference
score between heads, and expand the heads to its span.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class MovieCoreference(nn.Module):
    """Movie screenplay coreference model
    """

    def __init__(self,
                 encoder_name: str,
                 parsetag_size: int,
                 postag_size: int,
                 nertag_size: int,
                 tag_embedding_size: int,
                 gru_nlayers: int,
                 gru_hidden_size: int,
                 gru_bidirectional: bool,
                 topk: int,
                 dropout: float) -> None:
        """Initializer for the movie screenplay coreference model.

        Args:
            encoder_name: Name of the huggingface encoder.
            parsetag_size: Size of the movie screenplay parse tag set.
            postag_size: Size of the part-of-speech tag set.
            nertag_size: Size of the named-entity tag set.
            tag_embedding_size: Size of the tag embeddings used in character
                head recognition.
            gru_nlayers: Number of gru layers.
            gru_hidden_size: Hidden size of the gru layers.
            gru_bidirectional: If true, use bidirectional gru layers.
            topk: Maximum number of preceding antecedents to retain after
                coarse scoring.
            dropout: dropout rate used by all submodules.
        """
        super().__init__()
        self.dropout = dropout

        # Word Encoder
        self.encoder = AutoModel.from_pretrained(encoder_name, 
                                                 add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, 
                                                       use_fast=True)
        word_embedding_size = self.encoder.config.hidden_size
        self.attn = nn.Linear(word_embedding_size, 1)
        self.word_dropout = nn.Dropout(self.dropout)

        # Character Recognizer
        gru_input_size = word_embedding_size + 3*tag_embedding_size + 2
        gru_output_size = gru_hidden_size * (1 + gru_bidirectional)
        self.parsetag_embedding = nn.Embedding(parsetag_size, 
                                               tag_embedding_size)
        self.postag_embedding = nn.Embedding(postag_size, tag_embedding_size)
        self.nertag_embedding = nn.Embedding(nertag_size, tag_embedding_size)
        self.gru = nn.GRU(gru_input_size, gru_hidden_size,
                          num_layers=gru_nlayers, batch_first=True,
                          bidirectional=gru_bidirectional)
        self.gru_dropout = nn.Dropout(self.dropout)
        self.character_recognizer = nn.Linear(gru_output_size, 1)

        # Coarse Coreference Scorer
        self.topk = topk
        self.coarse_scorer = nn.Linear(word_embedding_size, word_embedding_size)
        self.coarse_dropout = nn.Dropout(self.dropout)

        # Pairwise Encoder
        self.genre_embedding = nn.Embedding(7, 20)
        self.distance_embedding = nn.Embedding(9, 20)
        self.speaker_embedding = nn.Embedding(2, 20)

        # Fine/Anaphoricity Coreference Scorer
        pairwise_encoding_size = 3*word_embedding_size + 3*20
        self.anaphoricity_scorer = nn.Sequential(
            nn.Linear(pairwise_encoding_size, word_embedding_size), 
            nn.LeakyReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(word_embedding_size, 1))

        # Span Predictor
        self.ffn = nn.Sequential(
            nn.Linear(word_embedding_size * 2 + 64, word_embedding_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(word_embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(64, 4, 3, 1, 1),
            nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.sp_distance_embedding = nn.Embedding(128, 64)
        self._device = torch.device("cpu")
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self._device = device
    
    def _rename_keys(self, 
                     weights_dict: dict[str, torch.Tensor], 
                     renaming_schema: dict[str, str]
                     ) -> dict[str, torch.Tensor]:
        """Rename the keys in the weights dictionary according to the 
        renaming schema.

        Args:
            weights_dict: Dictionary of tensors.
            renaming_schema: Dictionary of old keys to new keys.
        
        Return:
            New weights dictionary with renamed keys.
        """
        for old_key, new_key in renaming_schema.items():
            weights_dict[new_key] = weights_dict.pop(old_key)
        return weights_dict

    def load_weights(self, weights_path):
        """Load weights from trained wl-roberta model
        """
        weights = torch.load(weights_path)
        self.encoder.load_state_dict(weights["bert"], strict=False)
        self.attn.load_state_dict(self._rename_keys(
            weights["we"], {"attn.weight": "weight", "attn.bias": "bias"}))
        self.coarse_scorer.load_state_dict(self._rename_keys(
            weights["rough_scorer"], 
            {"bilinear.weight": "weight", "bilinear.bias": "bias"}))
        self.genre_embedding.load_state_dict(self._rename_keys(
            weights["pw"], {"genre_emb.weight": "weight"}), strict=False)
        self.distance_embedding.load_state_dict(self._rename_keys(
            weights["pw"], {"distance_emb.weight": "weight"}), strict=False)
        self.speaker_embedding.load_state_dict(self._rename_keys(
            weights["pw"], {"speaker_emb.weight": "weight"}), strict=False)
        self.anaphoricity_scorer.load_state_dict(self._rename_keys(
            weights["a_scorer"], 
            {"hidden.0.weight": "0.weight", "hidden.0.bias": "0.bias",
             "out.weight": "3.weight", "out.bias": "3.bias"}))
        self.ffn.load_state_dict(self._rename_keys(
            weights["sp"], 
            {"ffnn.0.weight": "0.weight", "ffnn.0.bias": "0.bias",
             "ffnn.3.weight": "3.weight", "ffnn.3.bias": "3.bias",
             "ffnn.6.weight": "6.weight", "ffnn.6.bias": "6.bias"}), 
             strict=False)
        self.conv.load_state_dict(self._rename_keys(
            weights["sp"], 
            {"conv.0.weight": "0.weight", "conv.0.bias": "0.bias",
             "conv.1.weight": "1.weight", "conv.1.bias": "1.bias"}), 
             strict=False)
        self.sp_distance_embedding.load_state_dict(self._rename_keys(
            weights["sp"], {"emb.weight": "weight"}), strict=False)
    
    def encode_words(self, 
                     subword_embeddings: torch.Tensor, 
                     word_to_subword_offset: torch.Tensor) -> torch.Tensor:
        """Find word embeddings from subword embeddings by attention-weighted
        linear combination.

        Args:
            subword_embeddings: [n_subwords, embedding_size] Subword embeddings
                of a document.
            word_to_subword_offset: [n_words, 2] Subword offsets of words. The
                end is inclusive.
        
        Returns:
            Word embeddings [n_words, embedding_size]
        """
        n_subwords = len(subword_embeddings)
        n_words = len(word_to_subword_offset)

        # attn_mask: [n_words, n_subwords]
        # with 0 at positions belonging to the words and -inf elsewhere
        attn_mask = torch.arange(
            0, n_subwords, device=self.device).expand((n_words, n_subwords))
        attn_mask = ((attn_mask >= word_to_subword_offset[:,0].unsqueeze(1))
                     * (attn_mask <= word_to_subword_offset[:,1].unsqueeze(1)))
        attn_mask = torch.log(attn_mask.to(torch.float))

        # attn_scores: [n_words, n_subwords]
        # with attention weights at positions belonging to the words and 0 
        # elsewhere
        attn_scores = self.attn(subword_embeddings).T
        attn_scores = attn_scores.expand((n_words, n_subwords))
        attn_scores = attn_mask + attn_scores
        del attn_mask
        attn_scores = torch.softmax(attn_scores, dim=1)

        # word_embeddings: [n_words, embedding_size]
        word_embeddings = attn_scores.mm(subword_embeddings)
        word_embeddings = self.word_dropout(word_embeddings)
        return word_embeddings
    
    def recognize_character_heads(self, 
                                  word_embeddings: torch.Tensor,
                                  parsetag_ids: torch.Tensor,
                                  postag_ids: torch.Tensor,
                                  nertag_ids: torch.Tensor,
                                  is_pronoun: torch.Tensor,
                                  is_punctuation: torch.Tensor
                                  ) -> torch.Tensor:
        """Find word scores of the word being the head of a character mention.

        Args:
            word_embeddings: [n_sequences, n_sequence_words, embedding_size]
                Batch of sequences of word embeddings.
            parsetag_ids: [n_sequences, n_sequence_words] Batch of sequences of
                movie screenplay tag ids.
            postag_ids: [n_sequences, n_sequence_words] Batch of sequences of
                word part-of-speech ids.
            nertag_ids: [n_sequences, n_sequence_words] Batch of sequences of
                word ner-tag ids.
            is_pronoun: [n_sequences, n_sequence_words] Batch of sequences of
                word is-pronoun flags. 1 if word is a pronoun, else 0.
            is_punctuation: [n_sequences, n_sequence_words] Batch of sequences
                of word is-punctuation flags. 1 if word is a punctuation, 
                else 0.
        
        Returns:
            Word scores of the word being the head of a character mention
            [n_sequences, n_sequence_words]
        """
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
        
        # character_scores: [n_sequences, n_sequence_words]
        gru_output, _ = self.gru(gru_input)
        gru_output = self.gru_dropout(gru_output)
        character_scores = self.character_recognizer(gru_output).squeeze(dim=2)
        return character_scores
    
    def coarse_coreference(self, 
                           word_embeddings: torch.Tensor, 
                           word_character_scores: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find coarse coreference scores of word pairs and return word ids of
        the topk scoring antecedents and their scores for each word.

        Args:
            word_embeddings: [n_words, embedding_size] Word embeddings.
            word_character_scores: [n_words] Word scores of the word being
                the head of a character mention.
        
        Returns:

        """
        