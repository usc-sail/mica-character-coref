"""Train wl-roberta model on movie screenplay coreference data in a distributed
system using huggingface accelerate library.
"""
from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference
from mica_text_coref.coref.movie_coref.data import CorefCorpus, CorefDocument, Mention
from mica_text_coref.coref.movie_coref.data import parse_labelset, pos_labelset, ner_labelset

import accelerate
from accelerate import logging
import bisect
import collections
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
        freeze_bert: bool,
        genre: str,
        bce_weight: float,
        bert_lr: float,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        document_len: int,
        overlap_len: int,
        cr_seq_len: int,
        subword_batch_size: int,
        cr_batch_size: int,
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
        self.freeze_bert = freeze_bert
        self.genre = genre
        self.bce_weight = bce_weight
        self.bert_lr = bert_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.document_len = document_len
        self.overlap_len = overlap_len
        self.cr_seq_len = cr_seq_len
        self.subword_batch_size = subword_batch_size
        self.cr_batch_size = cr_batch_size
        self.fn_batch_size = fn_batch_size
        self.save_model = save_model
        self.save_output = save_output
        self.save_loss_curve = save_loss_curve
        self.debug = debug
        self._log_vars(locals())
    
    def _add_log_file(self, log_file: str):
        if self.accelerator.is_local_main_process:
            file_handler = _logging.FileHandler(log_file, mode="w")
            self.logger.logger.addHandler(file_handler)
    
    def _debug(self, message: str):
        if self.debug:
            self.logger.info(message)
    
    def _debug_all(self, message: str):
        if self.debug:
            self.logger.info(message, main_process_only=False)
    
    def _log(self, message: str):
        self.logger.info(message)
    
    def _log_all(self, message: str):
        self.logger.info(message, main_process_only=False)
    
    def _log_vars(self, argvar: dict[str, any]):
        for arg, value in argvar.items():
            self._log(f"{arg:20s} = {value}")
    
    def _gpu_usage(self):
        desc = []
        for gpu in gpustat.new_query().gpus:
            desc.append(
                f"GPU {gpu.index} = {gpu.memory_used}/{gpu.memory_total}")
        return ", ".join(desc)

    def _split_screenplay(self, document: CorefDocument) -> list[CorefDocument]:
        documents: list[CorefDocument] = []
        doc_offsets: list[tuple[int, int]] = []
        segment_boundaries = np.zeros(len(document.token), dtype=int)
        i = 0
        parse_tags = document.parse
        sentence_offsets = np.array(document.sentence_offsets)
        while i < len(document.token):
            if parse_tags[i] in "SNC":
                j = i + 1
                while j < len(document.token) and parse_tags[j] == parse_tags[i]:
                    j += 1
                segment_boundaries[i] = 1
                i = j
            else:
                i += 1
        segment_boundaries[0] = 1
        i = 0
        while i < len(document.token):
            j = min(i + self.document_len, len(document.token))
            if j < len(document.token):
                while j >= i and segment_boundaries[j] == 0:
                    j -= 1
                k = i + self.document_len - self.overlap_len
                while k >= i and segment_boundaries[k] == 0:
                    k -= 1
                nexti = k
            else:
                nexti = j
            assert i < nexti, "Document length is 0!"
            doc_offsets.append((i, j))
            i = nexti
        for k, (i, j) in enumerate(doc_offsets):
            _document = CorefDocument()
            _document.movie = document.movie + f"_{k + 1}"
            _document.rater = document.rater
            _document.token = document.token[i: j]
            _document.parse = document.parse[i: j]
            _document.parse_ids = [parse_labelset[x] for x in _document.parse]
            _document.pos = document.pos[i: j]
            _document.pos_ids = [pos_labelset[x] for x in _document.pos]
            _document.ner = document.ner[i: j]
            _document.ner_ids = [ner_labelset[x] for x in _document.ner]
            _document.is_pronoun = document.is_pronoun[i: j]
            _document.is_punctuation = document.is_punctuation[i: j]
            _document.speaker = document.speaker[i: j]
            si = np.nonzero(sentence_offsets[:,0] == i)[0][0]
            sj = np.nonzero(sentence_offsets[:,1] == j - 1)[0][0] + 1
            _document.sentence_offsets = (
                sentence_offsets[si: sj] - sentence_offsets[si, 0]).tolist()
            clusters: dict[str, set[Mention]] = collections.defaultdict(set)
            n_mentions = 0
            for character, mentions in document.clusters.items():
                for mention in mentions:
                    assert (
                        mention.end < i or 
                        i <= mention.begin <= mention.end < j or 
                        j <= mention.begin), (
                            "Mention crosses document boundaries")
                    if i <= mention.begin <= mention.end < j:
                        mention.begin -= i
                        mention.end -= i
                        mention.head -= i
                        clusters[character].add(mention)
                        n_mentions += 1
            _document.clusters = clusters
            _document.word_cluster_ids = document.word_cluster_ids[i: j]
            _document.word_head_ids = document.word_head_ids[i: j]
            self._debug(
                f"{_document.movie}: {len(_document.token)} words, "
                f"{n_mentions} mentions, {len(_document.clusters)} clusters")
            documents.append(_document)
        return documents

    def _tokenize_document(self, document: CorefDocument):
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
    
    def _create_subword_dataloader(self, document: CorefDocument):
        self._debug(f"{document.movie}: Creating subword dataloader")
        L = self.model.tokenizer.max_len_single_sentence
        pad_id = self.model.tokenizer.pad_token_id
        cls_id = self.model.tokenizer.cls_token_id
        sep_id = self.model.tokenizer.sep_token_id
        subword_id_seqs, subword_mask_seqs = [], []
        subword_ids = document.subword_ids
        offset = document.word_to_subword_offset
        ends = [end for _, end in offset]
        i = 0
        while i < len(subword_ids):
            j = min(i + L, len(subword_ids))
            l = bisect.bisect_left(ends, j)
            if j < ends[l]:
                j = ends[l - 1]
            subword_id_seqs.append(
                [cls_id] + subword_ids[i : j] + [sep_id] + 
                [pad_id] * (L - j + i))
            subword_mask_seqs.append(
                [0] + [1] * (j - i) + [0] * (L - j + i + 1))
            i = j
        self._debug(
            f"{document.movie}: {len(subword_id_seqs)} subword sequences, {L} "
            f"subwords/sequence")
        subword_id_tensor = torch.LongTensor(subword_id_seqs)
        subword_mask_tensor = torch.FloatTensor(subword_mask_seqs)
        document.subword_dataloader = self.accelerator.prepare_data_loader(
            DataLoader(TensorDataset(
                subword_id_tensor, subword_mask_tensor), 
                batch_size=self.subword_batch_size))
        batch_size = math.ceil(
            len(subword_id_seqs)/len(document.subword_dataloader))
        self._debug(
            f"{document.movie}: Subword batch size = {batch_size} "
            "sequences/batch")

    def _prepare_corpus(self, corpus: CorefCorpus) -> CorefCorpus:
        _corpus = CorefCorpus()
        for document in corpus:
            for _document in self._split_screenplay(document):
                self._tokenize_document(_document)
                self._create_subword_dataloader(_document)
                _corpus.documents.append(_document)
        return _corpus
    
    def _create_cr_dataloader(
        self, word_embeddings: torch.Tensor, 
        document: CorefDocument) -> DataLoader:
        seq_parsetag_ids = []
        seq_postag_ids = []
        seq_nertag_ids = []
        seq_ispronoun = []
        seq_ispunctuation = []
        n_seqs = math.ceil(len(word_embeddings)/self.cr_seq_len)
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        for i in range(n_seqs):
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
                _seq_parsetag_ids += [parse_labelset.other_id] * padding_size
                _seq_postag_ids += [pos_labelset.other_id] * padding_size
                _seq_nertag_ids += [ner_labelset.other_id] * padding_size
                _seq_ispronoun += [False] * padding_size
                _seq_ispunctuation += [False] * padding_size
            seq_parsetag_ids.append(_seq_parsetag_ids)
            seq_postag_ids.append(_seq_postag_ids)
            seq_nertag_ids.append(_seq_nertag_ids)
            seq_ispronoun.append(_seq_ispronoun)
            seq_ispunctuation.append(_seq_ispunctuation)
        tensor_word_embeddings = torch.cat(
            [word_embeddings, 
            torch.zeros(
                (n_seqs * self.cr_seq_len - len(word_embeddings), 
                word_embeddings.shape[1]), device=self.accelerator.device)], 
            dim=0).view((n_seqs, self.cr_seq_len, -1))
        tensor_parsetag_ids = torch.LongTensor(
            seq_parsetag_ids).to(self.accelerator.device)
        tensor_postag_ids = torch.LongTensor(
            seq_postag_ids).to(self.accelerator.device)
        tensor_nertag_ids = torch.LongTensor(
            seq_nertag_ids).to(self.accelerator.device)
        tensor_ispronoun = torch.LongTensor(
            seq_ispronoun).to(self.accelerator.device)
        tensor_ispunctuation = torch.LongTensor(
            seq_parsetag_ids).to(self.accelerator.device)
        self._debug(
            f"{prefix}: cr word_embeddings = {tensor_word_embeddings.shape} "
            f"({tensor_word_embeddings.device})")
        self._debug(
            f"{prefix}: cr parsetag_ids = {tensor_parsetag_ids.shape} "
            f"({tensor_parsetag_ids.device})")
        self._debug(
            f"{prefix}: cr postag_ids = {tensor_postag_ids.shape} "
            f"({tensor_postag_ids.device})")
        self._debug(
            f"{prefix}: cr nertag_ids = {tensor_nertag_ids.shape} "
            f"({tensor_nertag_ids.device})")
        self._debug(
            f"{prefix}: cr ispronoun = {tensor_ispronoun.shape} "
            f"({tensor_ispronoun.device})")
        self._debug(
            f"{prefix}: cr ispunctuation = {tensor_ispunctuation.shape} "
            f"({tensor_ispunctuation.device})")
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

    def _create_fine_dataloader(
        self, word_embeddings, features, scores, indices, document):
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        dataset = TensorDataset(word_embeddings, features, indices, scores)
        dataloader = DataLoader(dataset, batch_size=self.fn_batch_size)
        dataloader = self.accelerator.prepare_data_loader(dataloader)
        batch_size = math.ceil(len(word_embeddings)/len(dataloader))
        self._debug(f"{prefix}: FN batch size = {batch_size} word pairs/batch")
        return dataloader

    def _get_coref_ground_truth(
        self, cluster_ids: list[int], top_indices: torch.Tensor, 
        valid_pair_map: torch.Tensor) -> torch.Tensor:
        cluster_ids = torch.Tensor(cluster_ids).to(self.accelerator.device)
        y = cluster_ids[top_indices] * valid_pair_map
        y[y == 0] = -1
        dummy = torch.zeros(
            (len(y), 1), dtype=y.dtype, device=self.accelerator.device)
        y = torch.cat([dummy, y], dim=1)
        y = (y == cluster_ids.unsqueeze(1))
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    def _train(self, document: CorefDocument):
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"

        # Zero grad
        if not self.freeze_bert:
            self.bert_optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        # Bert
        subword_embeddings = []
        for batch in document.subword_dataloader:
            ids, mask = batch
            embeddings = self.model.bert(ids, mask).last_hidden_state
            embeddings, mask = self.accelerator.gather_for_metrics(
                (embeddings, mask))
            embeddings = embeddings[mask == 1]
            subword_embeddings.append(embeddings)
            self.accelerator.wait_for_everyone()
        self._debug(self._gpu_usage())
        subword_embeddings = torch.cat(subword_embeddings, dim=0)
        self._debug_all(
            f"{prefix}: subword_embeddings = {subword_embeddings.shape} "
            f"({subword_embeddings.device})")
        self._debug(self._gpu_usage())
        
        # Word Encoder
        word_to_subword_offset = torch.LongTensor(
            document.word_to_subword_offset).to(self.accelerator.device)
        word_embeddings = self.model.encoder(
            subword_embeddings, word_to_subword_offset)
        self._debug_all(
            f"{prefix}: word_embeddings = {word_embeddings.shape} "
            f"({word_embeddings.device})")
        self._debug(self._gpu_usage())

        # Character Scores
        cr_dataloader = self._create_cr_dataloader(word_embeddings, document)
        character_scores = []
        for batch in cr_dataloader:
            embeddings = batch[0]
            scores = self.model.character_recognizer(*batch)
            embeddings, scores = self.accelerator.gather_for_metrics(
                (embeddings, scores))
            scores = scores[~(embeddings == 0).all(dim=2)]
            character_scores.append(scores)
            self.accelerator.wait_for_everyone()
        self._debug(self._gpu_usage())
        character_scores = torch.cat(character_scores, dim=0)
        self._debug_all(
            f"{prefix}: character_scores = {character_scores.shape} "
            f"({character_scores.device})")
        self._debug(self._gpu_usage())

        # Coarse Coreference Scores
        coarse_scores, top_indices = self.model.coarse_scorer(
            word_embeddings, character_scores)
        self._debug_all(
            f"{prefix}: coarse_scores = {coarse_scores.shape}, top_indices = "
            f"{top_indices.shape} ({top_indices.device})")
        torch.cuda.empty_cache()
        self._debug(self._gpu_usage())

        # Pairwise Encoder
        features = self.model.pairwise_encoder(
            top_indices, document.speaker, self.genre)
        self._debug_all(
            f"{prefix}: features = {features.shape} ({features.device})")
        self._debug(self._gpu_usage())
        
        # Fine Coreference Scores
        fn_dataloader = self._create_fine_dataloader(
            word_embeddings, features, coarse_scores, top_indices, document)
        fine_scores = []
        for i, batch in enumerate(fn_dataloader):
            batch_prefix = f"{prefix}, Batch {i + 1}"
            scores = self.model.fine_scorer(word_embeddings, *batch)
            self._debug_all(
                f"{batch_prefix}: scores (before gather) = {scores.shape} "
                f"({scores.device})")
            scores = self.accelerator.gather_for_metrics(scores)
            self._debug_all(
                f"{batch_prefix}: scores (after gather) = {scores.shape} "
                f"({scores.device})")
            fine_scores.append(scores)
            self.accelerator.wait_for_everyone()
        fine_scores = torch.cat(fine_scores, dim=0)
        self._debug_all(
            f"{prefix}: fine_scores = {fine_scores.shape} "
            f"({fine_scores.device})")
        torch.cuda.empty_cache()
        self._debug(self._gpu_usage())

        # Coreference Loss
        coref_y = self._get_coref_ground_truth(
            document.word_cluster_ids, top_indices,
            (coarse_scores > float("-inf")))
        coref_loss = self.model.coref_loss(fine_scores, coref_y)
        torch.cuda.empty_cache()
        self._debug(f"{prefix}: coref_loss = {coref_loss:.4f}")
        self._debug(self._gpu_usage())

        # Backprop
        self.accelerator.backward(coref_loss)
        if not self.freeze_bert:
            self.bert_optimizer.step()
        self.optimizer.step()

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
            bce_weight = self.bce_weight,
            dropout = self.dropout)
        self.model.device = self.accelerator.device
        self._debug(self._gpu_usage())

        self._debug("Loading model weights from word-level-coref model")
        self.model.load_weights(self.weights_path)
        self._debug(self._gpu_usage())

        self._debug("Creating optimizers")
        if self.freeze_bert:
            for param in self.model.bert_parameters():
                param.requires_grad = False
        else:
            self.bert_optimizer = AdamW(
                self.model.bert_parameters(), lr=self.bert_lr, 
                weight_decay=self.weight_decay)
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._debug("Accelerating model and optimizers")
        for module in self.model.modules():
            if next(module.parameters()).requires_grad:
                module = self.accelerator.prepare_model(module)
            else:
                module.to(self.accelerator.device)
        if self.freeze_bert:
            self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        else:
            self.bert_optimizer, self.optimizer = self.accelerator.prepare(
                self.bert_optimizer, self.optimizer)

        self._debug("Loading training corpus")
        self.train_corpus = CorefCorpus(self.train_path)
        self.train_corpus = self._prepare_corpus(self.train_corpus)
        
        self._debug("Loading development corpus")
        self.dev_corpus = CorefCorpus(self.dev_path)
        self.dev_corpus = self._prepare_corpus(self.dev_corpus)

        self._log("Starting training")
        self.model.train()
        self._debug(self._gpu_usage())
        for self.epoch in range(self.max_epochs):
            # TODO randomize this
            inds = np.arange(len(self.train_corpus))
            for self.doc_index in inds:
                document = self.train_corpus[self.doc_index]
                if document.movie == "prestige_5":
                    self._log_all(
                        f"Epoch = {self.epoch:2d}, Movie = {document.movie}")
                    self._train(document)
                    torch.cuda.empty_cache()
                    self._debug(self._gpu_usage())
                    self.accelerator.wait_for_everyone()
            break