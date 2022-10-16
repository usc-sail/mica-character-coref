"""Train wl-roberta model on movie screenplay coreference data in a distributed
system using huggingface accelerate library.
"""
from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference
from mica_text_coref.coref.movie_coref.data import CorefCorpus, CorefDocument

import accelerate
from accelerate import logging
import bisect
import logging as _logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

class CoreferenceTrainer:

    def __init__(self,
                 log_file: str,
                 parsetag_size: int,
                 postag_size: int,
                 nertag_size: int,
                 tag_embedding_size: int,
                 gru_nlayers: int,
                 gru_hidden_size: int,
                 gru_bidirectional: bool,
                 topk: int,
                 dropout: float,
                 weights_path: str,
                 train_path: str,
                 dev_path: str,
                 lr: float,
                 weight_decay: float,
                 max_epochs: int,
                 encoding_batch_size: int,
                 save_model: bool,
                 save_output: bool,
                 save_loss_curve: bool
                ) -> None:
        self.accelerator = accelerate.Accelerator()
        self.logger = logging.get_logger("")
        self._add_log_file(log_file)
        self.parsetag_size = parsetag_size
        self.postag_size = postag_size
        self.nertag_size = nertag_size
        self.tag_embedding_size = tag_embedding_size
        self.gru_nlayers = gru_nlayers
        self.gru_hidden_size = gru_hidden_size
        self.gru_bidirectional = gru_bidirectional
        self.topk = topk
        self.dropout = dropout
        self.weights_path = weights_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.encoding_batch_size = encoding_batch_size
        self.save_model = save_model
        self.save_output = save_output
        self.save_loss_curve = save_loss_curve
    
    def _add_log_file(self, log_file: str):
        if self.accelerator.is_local_main_process:
            file_handler = _logging.FileHandler(log_file)
            self.logger.logger.addHandler(file_handler)

    def _tokenize_corpus(self, corpus: CorefCorpus):
        for document in corpus:
            words = document.token
            subword_ids = []
            word_to_subword_offset = []
            for word in words:
                _subwords = self.model.tokenizer_map.get(
                    word, self.tokenizer.tokenize(word))
                word_to_subword_offset.append(
                    [len(subword_ids), len(subword_ids) + len(_subwords)])
                subword_ids.extend(self.model.tokenizer.convert_tokens_to_ids(
                    _subwords))
            document["subword_ids"] = subword_ids
            document["word_to_subword_offset"] = np.array(word_to_subword_offset)
    
    def _create_encoding_dataloaders(self, corpus):
        self._tokenize_corpus(corpus)
        L = self.model.tokenizer.max_len_single_sentence
        pad_id = self.model.tokenizer.pad_token_id
        cls_id = self.model.tokenizer.cls_token_id
        sep_id = self.model.tokenizer.sep_token_id
        for document in corpus:
            subword_id_seqs, subword_mask_seqs, offset_seqs = [], [], []
            subword_ids = document["subword_ids"]
            offset = document["word_to_subword_offset"]
            ends = offset[:, 1]
            i, k, n = 0, 0, 0
            while i < len(subword_ids):
                j = min(i + L, len(subword_ids))
                l = bisect.bisect_left(ends, j)
                if j < ends[l]:
                    j = ends[l - 1]
                else:
                    l = l + 1
                subword_id_seqs.append(
                    [cls_id] + subword_ids[i : j] + [sep_id] + 
                    [pad_id] * (L - j + i))
                subword_mask_seqs.append([1] * (j - i + 2) + [0] * (L - j + i))
                offset_seqs.append((offset[k: l] - offset[k, 0]).tolist())
                n = max(n, l - k)
                i = j
                k = l
            offset_seqs = [_seq + [[0, 0]] * (n - len(_seq)) 
                           for _seq in offset_seqs]
            subword_id_tensors = torch.LongTensor(subword_id_seqs)
            subword_mask_tensors = torch.FloatTensor(subword_mask_seqs)
            offset_tensors = torch.LongTensor(offset_seqs)
            document["subword_dataloader"] = self.accelerator.prepare(
                DataLoader(TensorDataset(
                    subword_id_tensors, subword_mask_tensors, offset_tensors), 
                    batch_size=self.encoding_batch_size))

    def run(self):
        self.model = MovieCoreference(
            parsetag_size = self.parsetag_size,
            postag_size = self.postag_size,
            nertag_size = self.nertag_size,
            tag_embedding_size = self.tag_embedding_size,
            gru_nlayers = self.gru_nlayers,
            gru_hidden_size = self.gru_hidden_size,
            gru_bidirectional = self.gru_bidirectional,
            topk = self.topk,
            dropout = self.dropout)
        self.model.device = self.accelerator.device
        self.model.load_weights(self.weights_path)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, 
                               weight_decay=self.weight_decay)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer)

        self.train_corpus = CorefCorpus(self.train_path)
        self.dev_corpus = CorefCorpus(self.dev_path)
        self._create_encoding_dataloaders(self.train_corpus)
        self._create_encoding_dataloaders(self.dev_corpus)

        self.model.train()
        for self.epoch in range(self.max_epochs):
            inds = np.random.permutation(len(self.train_corpus))
            for self.doc_index in inds:
                document = self.train_corpus[self.doc_index]