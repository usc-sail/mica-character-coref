"""Train wl-roberta model on movie screenplay coreference data in a distributed
system using huggingface accelerate library.
"""
from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference
from mica_text_coref.coref.movie_coref.data import CorefCorpus, CorefDocument
from mica_text_coref.coref.movie_coref.data import parse_labelset, pos_labelset, ner_labelset

import accelerate
from accelerate import logging
import bisect
import gpustat
import logging as _logging
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

class CoreferenceTrainer:

    def __init__(
        self,
        log_file: str,
        tag_embedding_size: int,
        gru_nlayers: int,
        gru_hidden_size: int,
        gru_bidirectional: bool,
        topk: int,
        dropout: float,
        weights_path: str,
        train_path: str,
        dev_path: str,
        bert_lr: float,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        cr_seq_len: int,
        subword_batch_size: int,
        cr_batch_size: int,
        cs_batch_size: int,
        fn_batch_size: int,
        save_model: bool,
        save_output: bool,
        save_loss_curve: bool,
        debug: bool
        ) -> None:
        self.accelerator = accelerate.Accelerator()
        self.logger = logging.get_logger("")
        self._add_log_file(log_file)
        self.tag_embedding_size = tag_embedding_size
        self.gru_nlayers = gru_nlayers
        self.gru_hidden_size = gru_hidden_size
        self.gru_bidirectional = gru_bidirectional
        self.topk = topk
        self.dropout = dropout
        self.weights_path = weights_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.bert_lr = bert_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.cr_seq_len = cr_seq_len
        self.subword_batch_size = subword_batch_size
        self.cr_batch_size = cr_batch_size
        self.cs_batch_size = cs_batch_size
        self.fn_batch_size = fn_batch_size
        self.save_model = save_model
        self.save_output = save_output
        self.save_loss_curve = save_loss_curve
        self.debug = debug
        self._log_vars(locals())
    
    def _add_log_file(self, log_file: str):
        if self.accelerator.is_local_main_process:
            file_handler = _logging.FileHandler(log_file)
            self.logger.logger.addHandler(file_handler)
    
    def _debug(self, message: str):
        if self.debug:
            self.logger.info(message)
    
    def _log(self, message: str):
        self.logger.info(message)
    
    def _log_vars(self, argvar: dict[str, any]):
        for arg, value in argvar.items():
            self._log(f"{arg:20s} = {value}")
    
    def _gpu_usage(self):
        desc = []
        for gpu in gpustat.new_query().gpus:
            desc.append(
                f"GPU {gpu.index} = {gpu.memory_used}/{gpu.memory_total}")
        return ", ".join(desc)

    def _tokenize_corpus(self, document: CorefDocument):
        self._debug(f"{document.movie}: Tokenizing screenplay")
        words = document.token
        subword_ids = []
        word_to_subword_offset = []
        for word in words:
            _subwords = self.model.tokenizer_map.get(
                word, self.model.tokenizer.tokenize(word))
            word_to_subword_offset.append(
                [len(subword_ids), len(subword_ids) + len(_subwords)])
            subword_ids.extend(self.model.tokenizer.convert_tokens_to_ids(
                _subwords))
        document.subword_ids = subword_ids
        document.word_to_subword_offset = word_to_subword_offset
        self._debug(
            f"{document.movie}: {len(subword_ids)} subwords, "
            f"{len(word_to_subword_offset)} words")
        self._debug(self._gpu_usage())
    
    def _create_subword_dataloader(self, document: CorefDocument):
        self._debug(f"{document.movie}: Creating subword dataloader")
        L = self.model.tokenizer.max_len_single_sentence
        pad_id = self.model.tokenizer.pad_token_id
        cls_id = self.model.tokenizer.cls_token_id
        sep_id = self.model.tokenizer.sep_token_id
        subword_id_seqs, subword_mask_seqs, offset_seqs = [], [], []
        subword_ids = document.subword_ids
        offset = document.word_to_subword_offset
        ends = [end for _, end in offset]
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
            offset_seqs.append(offset[k: l])
            n = max(n, l - k)
            i = j
            k = l
        self._debug(
            f"{document.movie}: {len(subword_id_seqs)} subword sequences, {L} "
            f"subwords/sequence, maximum {n} words/sequences")
        self._debug(self._gpu_usage())
        offset_seqs = [
            _seq + [[0, 0]] * (n - len(_seq)) for _seq in offset_seqs]
        subword_id_tensor = torch.LongTensor(subword_id_seqs)
        subword_mask_tensor = torch.FloatTensor(subword_mask_seqs)
        offset_tensor = torch.LongTensor(offset_seqs)
        document.subword_dataloader = self.accelerator.prepare_data_loader(
            DataLoader(TensorDataset(
                subword_id_tensor, subword_mask_tensor, offset_tensor), 
                batch_size=self.subword_batch_size))
        batch_size = math.ceil(
            len(subword_id_seqs)/len(document.subword_dataloader))
        self._debug(
            f"{document.movie}: Subword batch size = {batch_size} "
            "sequences/batch")
        self._debug(self._gpu_usage())

    def _prepare_corpus(self, corpus: CorefCorpus):
        for document in corpus:
            self._tokenize_corpus(document)
            self._create_subword_dataloader(document)
    
    def _create_cr_dataloader(
        self, word_embeddings: torch.Tensor, document: CorefDocument) -> DataLoader:
        seq_word_embeddings = []
        seq_parsetag_ids = []
        seq_postag_ids = []
        seq_nertag_ids = []
        seq_ispronoun = []
        seq_ispunctuation = []
        n_seqs = math.ceil(len(word_embeddings)/self.cr_seq_len)
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        for i in range(n_seqs):
            _seq_word_embeddings = word_embeddings[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_parsetag_ids = document.parse_ids[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_postag_ids = document.pos_ids[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_nertag_ids = document.ner_ids[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispronoun = document.is_pronoun[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispunctuation = document.is_punctuation[
                i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            if i == n_seqs - 1 and (
                len(word_embeddings) < (i + 1) * self.cr_seq_len):
                padding_size = (i + 1) * self.cr_seq_len - len(word_embeddings)
                _seq_word_embeddings = torch.cat(
                    [_seq_word_embeddings, 
                    torch.zeros(
                        padding_size, word_embeddings.shape[1], 
                        device=self.accelerator.device)], dim=0)
                _seq_parsetag_ids += [parse_labelset.other_id] * padding_size
                _seq_postag_ids += [pos_labelset.other_id] * padding_size
                _seq_nertag_ids += [ner_labelset.other_id] * padding_size
                _seq_ispronoun += [False] * padding_size
                _seq_ispunctuation += [False] * padding_size
            seq_word_embeddings.append(_seq_word_embeddings)
            seq_parsetag_ids.append(_seq_parsetag_ids)
            seq_postag_ids.append(_seq_postag_ids)
            seq_nertag_ids.append(_seq_nertag_ids)
            seq_ispronoun.append(_seq_ispronoun)
            seq_ispunctuation.append(_seq_ispunctuation)
        tensor_word_embeddings = torch.cat(seq_word_embeddings, dim=0)
        tensor_parsetag_ids = torch.LongTensor(
            seq_parsetag_ids, device=self.accelerator.device)
        tensor_postag_ids = torch.LongTensor(
            seq_postag_ids, device=self.accelerator.device)
        tensor_nertag_ids = torch.LongTensor(
            seq_nertag_ids, device=self.accelerator.device)
        tensor_ispronoun = torch.LongTensor(
            seq_ispronoun, device=self.accelerator.device)
        tensor_ispunctuation = torch.LongTensor(
            seq_parsetag_ids, device=self.accelerator.device)
        self._debug(
            f"{prefix}: {len(tensor_word_embeddings)} word embedding "
            f"sequences, {tensor_word_embeddings.shape[1]} words/sequence")
        dataloader = self.accelerator.prepare_data_loader(DataLoader(
            TensorDataset(
                tensor_word_embeddings, 
                tensor_parsetag_ids, 
                tensor_postag_ids, 
                tensor_nertag_ids, 
                tensor_ispronoun, 
                tensor_ispunctuation),
            batch_size=self.cr_batch_size))
        batch_size = math.ceil(
            len(tensor_word_embeddings)/len(dataloader))
        self._debug(
            f"{prefix}: CR batch size = {batch_size} sequences/batch")
        return dataloader

    def _train(self, document: CorefDocument):
        word_embeddings = []
        character_scores = []
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        self._debug(f"{prefix}: {self._gpu_usage()}")
        for i, batch in enumerate(document.subword_dataloader):
            batch_prefix = f"{prefix}, Batch {i + 1:2d}"
            batch_word_embeddings = self.model.encoder(*batch)
            batch_word_embeddings = self.accelerator.gather_for_metrics(
                batch_word_embeddings)
            self._debug(
                f"{batch_prefix}: batch word embeddings = "
                f"{batch_word_embeddings.shape} "
                f"({batch_word_embeddings.device})")
            word_embeddings.append(batch_word_embeddings)
            self._debug(f"{batch_prefix}: {self._gpu_usage()}")
        self._debug(f"{prefix}: {self._gpu_usage()}")
        word_embeddings = torch.cat(word_embeddings, dim=0)
        self._debug(
            f"{prefix}: word embeddings = {word_embeddings.shape} "
            f"({word_embeddings.device})")
        self._debug(f"{prefix}: {self._gpu_usage()}")
        cr_dataloader = self._create_cr_dataloader(word_embeddings, document)
        for i, batch in enumerate(cr_dataloader):
            batch_prefix = f"{prefix}, Batch {i + 1:2d}"
            batch_character_scores = self.model.character_recognizer(*batch)
            batch_character_scores = self.accelerator.gather_for_metrics(
                batch_character_scores)
            self._debug(
                f"{prefix}: batch character scores = "
                f"{batch_character_scores.shape} "
                f"({batch_character_scores.device})")
            character_scores.append(batch_character_scores)
            self._debug(f"{batch_prefix}: {self._gpu_usage()}")
        self._debug(f"{prefix}: {self._gpu_usage()}")
        character_scores = torch.cat(character_scores, dim=0)
        self._debug(
            f"{prefix}: character scores = {character_scores.shape} "
            f"({character_scores.device})")
        self._debug(f"{prefix}: {self._gpu_usage()}")

    def run(self):
        self._debug(self._gpu_usage())
        self._debug("Initializing model")
        self.model = MovieCoreference(
            parsetag_size = len(parse_labelset),
            postag_size = len(pos_labelset),
            nertag_size = len(ner_labelset),
            tag_embedding_size = self.tag_embedding_size,
            gru_nlayers = self.gru_nlayers,
            gru_hidden_size = self.gru_hidden_size,
            gru_bidirectional = self.gru_bidirectional,
            topk = self.topk,
            dropout = self.dropout)
        self.model.device = self.accelerator.device
        self._debug(self._gpu_usage())

        self._debug("Loading model weights from word-level-coref model")
        self.model.load_weights(self.weights_path)
        self._debug(self._gpu_usage())

        self._debug("Creating optimizers")
        self.bert_optimizer = AdamW(
            self.model.bert_parameters(), lr=self.bert_lr, 
            weight_decay=self.weight_decay)
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._debug(self._gpu_usage())

        self._debug("Accelerating model and optimizers")
        for module in self.model.modules():
            module = self.accelerator.prepare_model(module)
        self.bert_optimizer, self.optimizer = self.accelerator.prepare(
            self.bert_optimizer, self.optimizer)
        self._debug(self._gpu_usage())

        self._debug("Loading training corpus")
        self.train_corpus = CorefCorpus(self.train_path)
        self._prepare_corpus(self.train_corpus)
        self._debug(self._gpu_usage())
        
        self._debug("Loading development corpus")
        self.dev_corpus = CorefCorpus(self.dev_path)
        self._prepare_corpus(self.dev_corpus)
        self._debug(self._gpu_usage())

        self._log("Starting training")
        self.model.train()
        self._debug(self._gpu_usage())
        for self.epoch in range(self.max_epochs):
            inds = np.random.permutation(len(self.train_corpus))
            for self.doc_index in inds:
                document = self.train_corpus[self.doc_index]
                self._log(f"Epoch = {self.epoch:2d}, Movie = {document.movie}")
                self._train(document)