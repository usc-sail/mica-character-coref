"""Train movie coreference model on single GPU"""
from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference
from mica_text_coref.coref.movie_coref.data import CorefCorpus, CorefDocument, Mention, GraphNode
from mica_text_coref.coref.movie_coref.data import parse_labelset, pos_labelset, ner_labelset
from mica_text_coref.coref.movie_coref.result import CorefResult

import bisect
import collections
import gpustat
import itertools
import jsonlines
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import tqdm
import yaml

value = 2023
random.seed(value)
np.random.seed(value)
torch.manual_seed(value)
torch.cuda.manual_seed_all(value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  

class CoreferenceTrainer:
    """Train movie coreference model"""

    def __init__(
        self,
        preprocess: str,
        output_dir: str,
        reference_scorer_file: str,
        train_file: str,
        dev_file: str,
        weights_file: str,
        tag_embedding_size: int = 16,
        gru_nlayers: int = 1,
        gru_hidden_size: int = 256,
        gru_bidirectional: bool = True,
        topk: int = 50,
        dropout: float = 0,
        freeze_bert: bool = False,
        genre: str = "wb",
        bce_weight: float = 0.5,
        bert_lr: float = 1e-5,
        character_lr: float = 1e-4,
        coref_lr: float = 1e-4,
        warmup_epochs: float = None,
        weight_decay: float = 0,
        max_epochs: int = 20,
        patience: int = 3,
        train_document_len: int = 5120,
        subword_batch_size: int = 8,
        cr_seq_len: int = 256,
        cr_batch_size: int = 16,
        fn_batch_size: int = 16,
        sp_batch_size: int = 16,
        evaluate_train: bool = False,
        n_epochs_no_eval: int = 0,
        add_cr_to_coarse: bool = True,
        filter_mentions_by_cr: bool = False,
        remove_singleton_cr: bool = True,
        save_log: bool = True,
        save_model: bool = False,
        save_predictions: bool = True,
        save_loss_curve: bool = True
        ) -> None:
        """Initialize coreference trainer

        Args:
            preprocess: type of preprocessing used on screenplays
            output_dir: directory where predictions, models, plots, and logs will be written
            reference_scorer_file: file path of the official conll-2012 coreference scorer
            train_file: file path of the training screenplays
            dev_file: file path of the development screenplays
            weights_file: file path of the pretrained word-level coreference model
            tag_embedding_size: embedding size of parse, pos, and ner tags in the character recognition model
            gru_nlayers: number of gru layers in the character recognition model
            gru_hidden_size: gru hidden size in the character recognition model
            gru_bidirectional: if true, use bidirectional layers in the gru of the character recognition model
            topk: number of top-scoring antecedents to retain for each word after coarse coreference scoring
            dropout: model-wide dropout probability
            freeze_bert: if true, roberta transformer is not trained
            genre: genre to use for pretrained word-level coreference model
            bce_weight: weight of binary cross entropy coreference loss
            bert_lr: learning rate of the roberta transformer
            character_lr: learning rate of the character recognition model
            coref_lr: learning rate of the coreference modules except the roberta transformer
            warmup_epochs: number of epochs for the learning rate to ramp up, can be a fraction
            weight_decay: l2 regularization weight to use in Adam optimizer
            max_epochs: maximum number of epochs to train the model
            patience: maximum number of epochs to wait for dev performance to improve before early stopping
            train_document_len: size of the train subdocuments in words
            subword_batch_size: number of subword sequences in batch
            cr_seq_len: sequence length for character recognizer model
            cr_batch_size: number of sequences in a character recognition batch
            fn_batch_size: number of word pairs in fine scoring batch
            sp_batch_size: number of heads in span prediction batch
            evaluate_train: if true, evaluate training set as well
            n_epochs_no_eval: initial number of epochs for which evaluation is not done, set it greater than zero if the model is not stable in the initial few epochs
            add_cr_to_coarse: if true, add character head scores to coarse scores
            filter_mentions_by_cr: if true, only keep words whose character score is positive while constructing the word clusters during inference
            remove_singleton_cr: if true, remove singleton word clusters during inference
            save_log: if true, save logs to file
            save_model: if true, save model weights of the epoch with the best dev performance
            save_predictions: if true, save model predictions of the epoch with the best dev performance
            save_loss_curve: if true, save train and dev loss curve
        """
        # training vars
        self.preprocess = preprocess
        self.output_dir = output_dir
        self.reference_scorer_file = reference_scorer_file
        self.train_file = train_file
        self.dev_file = dev_file
        self.weights_file = weights_file
        self.tag_embedding_size = tag_embedding_size
        self.gru_nlayers = gru_nlayers
        self.gru_hidden_size = gru_hidden_size
        self.gru_bidirectional = gru_bidirectional
        self.topk = topk
        self.dropout = dropout
        self.freeze_bert = freeze_bert
        self.genre = genre
        self.bce_weight = bce_weight
        self.bert_lr = bert_lr
        self.character_lr = character_lr
        self.coref_lr = coref_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_document_len = train_document_len
        self.subword_batch_size = subword_batch_size
        self.cr_seq_len = cr_seq_len
        self.cr_batch_size = cr_batch_size
        self.fn_batch_size = fn_batch_size
        self.sp_batch_size = sp_batch_size
        self.evaluate_train = evaluate_train
        self.n_epochs_no_eval = n_epochs_no_eval
        self.add_cr_to_coarse = add_cr_to_coarse
        self.filter_mentions_by_cr = filter_mentions_by_cr
        self.remove_singleton_cr = remove_singleton_cr
        self.save_log = save_log
        self.save_model = save_model
        self.save_predictions = save_predictions
        self.save_loss_curve = save_loss_curve

        # output dir
        if save_log or save_model or save_predictions or save_loss_curve:
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
        
        # logging
        self.logger = logging.getLogger("")
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.handlers = []
        self.logger.addHandler(console_handler)
        if save_log:
            self._add_log_file(os.path.join(self.output_dir, "train.log"))
        
        # log training vars
        self._log_vars(locals())

        # cuda device
        self.device = "cuda:0"

    def _add_log_file(self, log_file: str):
        """Add file handler to logger"""
        file_handler = logging.FileHandler(log_file, mode="w")
        self.logger.addHandler(file_handler)

    def _log(self, message: str):
        """Logger"""
        self.logger.info(message)

    def _log_vars(self, argvar: dict[str, any]):
        """Log training vars"""
        for arg, value in argvar.items():
            self._log(f"{arg:20s} = {value}")

    def _gpu_usage(self):
        """Log GPU usage"""
        desc = []
        for gpu in gpustat.new_query().gpus:
            desc.append(f"GPU {gpu.index} = {gpu.memory_used}/{gpu.memory_total}")
        return ", ".join(desc)

    def _create_optimizers(self):
        """Create optimizers. If freeze_bert is true, don't create the bert optimizer"""
        if self.freeze_bert:
            for param in self.model.bert_parameters():
                param.requires_grad = False
        else:
            self.bert_optimizer = AdamW(self.model.bert_parameters(), lr=self.bert_lr, weight_decay=self.weight_decay)
            if self.warmup_epochs >= 0:
                self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_optimizer, self.warmup_epochs * len(self.train_corpus), len(self.train_corpus) * self.max_epochs)
        self.cr_optimizer = AdamW(self.model.cr_parameters(), lr=self.character_lr, weight_decay=self.weight_decay)
        self.coref_optimizer = AdamW(self.model.coref_parameters(), lr=self.coref_lr, weight_decay=self.weight_decay)
        if self.warmup_epochs >= 0:
            self.cr_scheduler = get_linear_schedule_with_warmup(self.cr_optimizer, self.warmup_epochs * len(self.train_corpus), len(self.train_corpus) * self.max_epochs)
            self.coref_scheduler = get_linear_schedule_with_warmup(self.coref_optimizer, self.warmup_epochs * len(self.train_corpus), len(self.train_corpus) * self.max_epochs)

    def _split_screenplay(self, document: CorefDocument, split_len: int, overlap_len: int, exclude_subdocuments_with_no_clusters: bool = True, verbose = False):
        """Split screenplay document into smaller documents

        Args:
            document: CorefDocument object representing the original screenplay document
            split_len:lLength of the smaller CorefDocument objects in words
            overlap_len: number of words overlapping between successive smaller CorefDocuments
            exclude_subdocuments_with_no_clusters: if true, exclude subdocuments if they contain no clusters
        
        Returns:
            Generator of CorefDocument objects.
        """
        n_words = len(document.token)
        n_mentions = sum([len(cluster) for cluster in document.clusters.values()])
        n_clusters = len(document.clusters)
        if verbose:
            self._log(f"{document.movie}: {n_words} words, {n_mentions} mentions, {n_clusters} clusters")
        doc_offsets: list[tuple[int, int]] = []
        sentence_offsets = np.array(document.sentence_offsets)

        # find segment boundaries
        segment_boundaries = np.zeros(len(document.token), dtype=int)
        i = 0
        parse_tags = document.parse
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

        # find subdocument offsets
        i = 0
        while i < len(document.token):
            j = min(i + split_len, len(document.token))
            if j < len(document.token):
                while j >= i and segment_boundaries[j] == 0:
                    j -= 1
                k = j - overlap_len
                while k >= i and segment_boundaries[k] == 0:
                    k -= 1
                nexti = k
            else:
                nexti = j
            assert i < nexti, "Document length is 0!"
            doc_offsets.append((i, j))
            i = nexti
        
        # split screenplay into subdocument according to offsets
        for k, (i, j) in enumerate(doc_offsets):
            _document = CorefDocument()

            # populate subdocument-length fields
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

            # populate sentence offsets
            si = np.nonzero(sentence_offsets[:,0] == i)[0][0]
            sj = np.nonzero(sentence_offsets[:,1] == j - 1)[0][0] + 1
            _document.sentence_offsets = (sentence_offsets[si: sj] - sentence_offsets[si, 0]).tolist()

            # populate clusters
            clusters: dict[str, set[Mention]] = collections.defaultdict(set)
            n_mentions = 0
            for character, mentions in document.clusters.items():
                for mention in mentions:
                    assert (mention.end < i or i <= mention.begin <= mention.end < j or j <= mention.begin), "Mention crosses subdocument boundaries"
                    if i <= mention.begin <= mention.end < j:
                        mention.begin -= i
                        mention.end -= i
                        mention.head -= i
                        clusters[character].add(mention)
                        n_mentions += 1
            
            # next document if clusters is empty
            if exclude_subdocuments_with_no_clusters and len(clusters) == 0:
                continue

            _document.clusters = clusters
            _document.word_cluster_ids = document.word_cluster_ids[i: j]
            _document.word_head_ids = document.word_head_ids[i: j]
            if verbose:
                self._log(f"{_document.movie}: {len(_document.token)} words, {n_mentions} mentions, {len(_document.clusters)} clusters")
            yield _document

    def _tokenize_document(self, document: CorefDocument, verbose = False):
        """Tokenize the words of the document into subwords"""
        words = document.token
        subword_ids = []
        word_to_subword_offset = []
        for word in words:
            _subwords = self.model.tokenizer_map.get(word, self.model.tokenizer.tokenize(word))
            word_to_subword_offset.append([len(subword_ids), len(subword_ids) + len(_subwords)])
            subword_ids.extend(self.model.tokenizer.convert_tokens_to_ids(_subwords))
        document.subword_ids = subword_ids
        document.word_to_subword_offset = word_to_subword_offset
        if verbose:
            self._log(f"{document.movie}: {len(words)} words, {len(subword_ids)} subwords")

    def _create_subword_dataloader(self, document: CorefDocument, verbose = False):
        """Create dataloader of subword sequences and mask."""
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
            subword_id_seqs.append([cls_id] + subword_ids[i : j] + [sep_id] + [pad_id] * (L - j + i))
            subword_mask_seqs.append([0] + [1] * (j - i) + [0] * (L - j + i + 1))
            i = j
        subword_id_tensor = torch.LongTensor(subword_id_seqs).to(self.device)
        subword_mask_tensor = torch.FloatTensor(subword_mask_seqs).to(self.device)
        document.subword_dataloader = DataLoader(TensorDataset(subword_id_tensor, subword_mask_tensor), batch_size=self.subword_batch_size)
        if verbose:
            self._log(f"{document.movie}: {len(subword_id_seqs)} subword sequences")

    def _prepare_corpus(self, corpus: CorefCorpus, split_len: int = None, overlap_len = 0, exclude_subdocuments_with_no_clusters: bool = True, verbose = False) -> CorefCorpus:
        """Create new corpus by splitting, tokenizing, and creating subword sequences of the screenplay documents in the original corpus

        Args:
            corpus: CorefCorpus containing screenplay documents
            split_len: length of the new subdocuments in words the screenplay documents will be split into, if split_len is None then no splitting occurs
            overlap_len: number of words overlapping between successive subdocuments, ignored if no splitting occurs
            exclude_subdocuments_with_no_clusters: if true, exclude subdocuments if they contain no clusters
        """
        _corpus = CorefCorpus()
        if split_len is None:
            document_generator = corpus
        else:
            document_generator = itertools.chain(*[self._split_screenplay(document, split_len, overlap_len, exclude_subdocuments_with_no_clusters=exclude_subdocuments_with_no_clusters, 
                verbose=verbose) for document in corpus])
        for _document in document_generator:
            self._tokenize_document(_document, verbose=verbose)
            self._create_subword_dataloader(_document, verbose=verbose)
            _corpus.documents.append(_document)
        return _corpus

    def _find_max_n_words_left_and_right_of_head(self, corpus: CorefCorpus) -> tuple[int, int]:
        """Get the maximum number of words to the left and right of a head word"""
        max_left, max_right = -1, -1
        for document in corpus:
            for mentions in document.clusters.values():
                for mention in mentions:
                    max_left = max(max_left, mention.head - mention.begin)
                    max_right = max(max_right, mention.end - mention.head)
        return max_left, max_right
    
    def _find_avg_n_heads(self, corpus: CorefCorpus) -> float:
        """Get average number of character mention heads per document."""
        n_heads = []
        for document in corpus:
            n_document_heads = 0
            for mentions in document.clusters.values():
                n_document_heads += len(mentions)
            n_heads.append(n_document_heads)
        return np.mean(n_heads)

    def _create_cr_dataloader(self, word_embeddings: torch.FloatTensor, document: CorefDocument, verbose = False) -> DataLoader:
        """Get dataloader for character head recognition model

        Args:
            word_embeddings: float tensor of word embeddings [n_words, embedding_size]
            document: CorefDocument
        
        Returns:
            Tensor Dataloader
        """
        # Initialize the batch variables
        seq_parsetag_ids = []
        seq_postag_ids = []
        seq_nertag_ids = []
        seq_ispronoun = []
        seq_ispunctuation = []
        n_seqs = math.ceil(len(word_embeddings)/self.cr_seq_len)

        # populate each sequence
        for i in range(n_seqs):
            _seq_parsetag_ids = document.parse_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_postag_ids = document.pos_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_nertag_ids = document.ner_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispronoun = document.is_pronoun[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispunctuation = document.is_punctuation[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            if i == n_seqs - 1 and (len(word_embeddings) < (i + 1) * self.cr_seq_len):
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
        
        # tensorize sequences
        tensor_word_embeddings = torch.cat([word_embeddings, torch.zeros((n_seqs * self.cr_seq_len - len(word_embeddings), word_embeddings.shape[1]), device=self.device)], dim=0).view(
            (n_seqs, self.cr_seq_len, -1))
        tensor_parsetag_ids = torch.LongTensor(seq_parsetag_ids).to(self.device)
        tensor_postag_ids = torch.LongTensor(seq_postag_ids).to(self.device)
        tensor_nertag_ids = torch.LongTensor(seq_nertag_ids).to(self.device)
        tensor_ispronoun = torch.LongTensor(seq_ispronoun).to(self.device)
        tensor_ispunctuation = torch.LongTensor(seq_parsetag_ids).to(self.device)
        dataloader = DataLoader(TensorDataset(tensor_word_embeddings, tensor_parsetag_ids, tensor_postag_ids, tensor_nertag_ids, tensor_ispronoun, tensor_ispunctuation), batch_size=self.cr_batch_size)
        batch_size = math.ceil(len(tensor_word_embeddings)/len(dataloader))
        if verbose:
            prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
            self._log(f"{prefix}: cr batch size = {batch_size} sequences/batch")
        return dataloader

    def _create_fine_dataloader(self, word_embeddings: torch.FloatTensor, features: torch.FloatTensor, scores: torch.FloatTensor, indices: torch.LongTensor, document: CorefDocument) -> DataLoader:
        """Get dataloader for fine coreference scoring

        Args:
            word_embeddings: float tensor of word embeddings [n_words, embedding_size]
            features: float tensor of pairwise features (genre, same speaker, and distance) of words and its top k scoring antecedents. [n_words, n_antecedents, feature_size]
            scores: float tensor of coarse coreference scores of words and its top k scoring antecedents. [n_words, n_antecedents]
            indices: long tensor of the word indices of the top scoring antecedents. [n_words, n_antecedents]. Lies between 0 and n_words - 1
            document: CorefDocument object.
        
        Returns:
            Torch dataloader.
        """
        dataset = TensorDataset(word_embeddings, features, indices, scores)
        dataloader = DataLoader(dataset, batch_size=self.fn_batch_size)
        return dataloader

    def _get_coref_ground_truth(self, cluster_ids: list[int], top_indices: torch.LongTensor, valid_pair_map: torch.BoolTensor) -> torch.FloatTensor:
        """Get coreference ground truth for evaluation

        Args:
            cluster_ids: List of word cluster ids. 0 if word is not a character head, >=1 if word is a character head. Co-referring words have same cluster id. [n_words]
            top_indices: Long tensor of top scoring antecedents. [n_words, n_antecedents]
            valid_pair_mask: Bool tensor of whether the word-antecedent pair is valid (word comes after antecedents). [n_words, n_antecedents]
        
        Returns:
            Coreference labels y. Float tensor of shape [n_words, n_antecedents + 1].
            y[i, j + 1] = 1 if the ith word co-refers with its jth antecedent else 0, for j = 0 to n_antecedents - 1
            y[i, 0] = 1 if for all j = 0 to n_antecedents - 1: y[i, j + 1] = 0, else 0
        """
        cluster_ids = torch.Tensor(cluster_ids).to(self.device)
        y = cluster_ids[top_indices] * valid_pair_map
        y[y == 0] = -1
        dummy = torch.zeros((len(y), 1), dtype=y.dtype, device=self.device)
        y = torch.cat([dummy, y], dim=1)
        y = (y == cluster_ids.unsqueeze(1))
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    def _get_word_clusters(self, document: CorefDocument, character_scores: torch.FloatTensor, coref_scores: torch.FloatTensor, indices: torch.LongTensor) -> tuple[list[set[int]], list[set[int]]]:
        """Find word-level clusters from character head and coreference scores. Return both gold and predicted clusters.

        Args:
            character_scores: Float tensor of word-level logits of the word being the head of a character mention. [n_words]
            coref_scores: Float tensor of logits of the word pair being coreferent with each other. The first column contains all zeros. [n_words, 1 + n_antecedents]
            indices: Long tensor of antecedent indices. [n_words, n_antecedents]
        
        Returns:
            list[set[int]], list[set[int]]
        """
        # find antecedents of words
        is_character = (character_scores > 0).tolist()
        antecedents = coref_scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(coref_scores))[not_dummy]
        antecedents = indices[coref_span_heads, antecedents[not_dummy]]

        # link word with antecedents in a graph
        nodes = [GraphNode(i) for i in range(len(coref_scores))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            if not self.filter_mentions_by_cr or (is_character[i] and is_character[j]):
                nodes[i].link(nodes[j])
                assert nodes[i] is not nodes[j]

        # find components in the graph and get the predicted clusters
        pred_clusters = []
        for node in nodes:
            if not node.visited and (not self.filter_mentions_by_cr or is_character[node.id]):
                cluster = set([])
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.add(current_node.id)
                    stack.extend(_node for _node in current_node.neighbors if not _node.visited)
                if not self.remove_singleton_cr or len(cluster) > 1:
                    pred_clusters.append(cluster)
        
        # check mentions don't repeat between clusters
        for i in range(len(pred_clusters)):
            for j in range(i + 1, len(pred_clusters)):
                assert pred_clusters[i].isdisjoint(pred_clusters[j]), f"mentions repeat in predicted cluster {pred_clusters[i]} and cluster {pred_clusters[j]}"

        # get gold clusters
        gold_clusters = []
        for mentions in document.clusters.values():
            cluster = set([])
            for mention in mentions:
                cluster.add(mention.head)
            gold_clusters.append(cluster)

        return gold_clusters, pred_clusters

    def _get_sp_training_data(self, document: CorefDocument) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Get head, start, and end word indexes from document for training the span prediction module.

        Returns:
            heads: Long tensor of word indices of heads of character mentions [n_heads].
            starts: Long tensor of word indices of beginning of character mentions [n_heads].
            ends: Long tensor of word indices of ending of character mentions [n_heads].
        """
        starts, ends, heads = [], [], []
        for mentions in document.clusters.values():
            for mention in mentions:
                starts.append(mention.begin)
                ends.append(mention.end)
                heads.append(mention.head)
        starts = torch.LongTensor(starts).to(self.device)
        ends = torch.LongTensor(ends).to(self.device)
        heads = torch.LongTensor(heads).to(self.device)
        return heads, starts, ends

    def _get_sp_dataloader(self, head_ids: torch.LongTensor) -> DataLoader:
        """Get head ids dataloader for span predictor module"""
        dataset = TensorDataset(head_ids)
        dataloader = DataLoader(dataset, batch_size=self.sp_batch_size)
        return dataloader
    
    def _get_sp_inference_data(self, word_clusters: list[list[int]]) -> torch.LongTensor:
        """Get heads from the predicted word clusters."""
        heads = set([_head for cluster in word_clusters for _head in cluster])
        heads = sorted(heads)
        heads = torch.LongTensor(heads).to(self.device)
        return heads
    
    def _remove_repeated_mentions(self, clusters: list[set]) -> int:
        """Remove repeated mentions"""
        n = 0
        cluster_count = []
        span2clusters = collections.defaultdict(set)
        for i, cluster in enumerate(clusters):
            for span in cluster:
                span2clusters[span].add(i)
            cluster_count.append(len(cluster))
        for span, _clusters in span2clusters.items():
            if len(_clusters) > 1:
                _clusters = sorted(_clusters, key=lambda c: cluster_count[c])
                for i in _clusters[1:]:
                    clusters[i].discard(span)
                    n += 1
        return n

    def _run(self, document: CorefDocument) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CorefResult]:
        """Model coreference in document.

        Returns:
            character_loss: character recognition loss
            coref_loss: coreference loss
            span_loss: span prediction loss
            coref_result: CorefResult object containing gold and predicted word clusters, character heads, and span clusters
        
        In evaluation, the span_loss term is zero
        """
        result = CorefResult(self.reference_scorer_file, self.output_dir, document.movie)

        # Bertify
        subword_embeddings = []
        for batch in document.subword_dataloader:
            ids, mask = batch
            embeddings = self.model.bert(ids, mask).last_hidden_state
            embeddings = embeddings[mask == 1]
            subword_embeddings.append(embeddings)
        subword_embeddings = torch.cat(subword_embeddings, dim=0)

        # Word Encoder
        word_to_subword_offset = torch.LongTensor(document.word_to_subword_offset).to(self.device)
        word_embeddings = self.model.encoder(subword_embeddings, word_to_subword_offset)

        # Character Scores
        character_scores = []
        cr_dataloader = self._create_cr_dataloader(word_embeddings, document)
        for batch in cr_dataloader:
            embeddings = batch[0]
            scores = self.model.character_recognizer(*batch)
            scores = scores[~(embeddings == 0).all(dim=2)]
            character_scores.append(scores)
        character_scores = torch.cat(character_scores, dim=0)

        # Coarse Coreference Scores
        if self.add_cr_to_coarse:
            coarse_scores, top_indices = self.model.coarse_scorer(word_embeddings, character_scores)
        else:
            coarse_scores, top_indices = self.model.coarse_scorer(word_embeddings)

        # Pairwise Encoder
        features = self.model.pairwise_encoder(top_indices, document.speaker, self.genre)
        
        # Fine Coreference Scores
        fine_scores = []
        fn_dataloader = self._create_fine_dataloader(word_embeddings, features, coarse_scores, top_indices, document)
        for batch in fn_dataloader:
            scores = self.model.fine_scorer(word_embeddings, *batch)
            fine_scores.append(scores)
        fine_scores = torch.cat(fine_scores, dim=0)

        # Loss
        coref_y = self._get_coref_ground_truth(document.word_cluster_ids, top_indices, (coarse_scores > float("-inf")))
        coref_loss = self.model.coref_loss(fine_scores, coref_y)
        character_y = torch.FloatTensor(document.word_head_ids).to(self.device)
        character_loss = self.model.cr_loss(character_scores, character_y)

        # Performance
        gold_word_clusters, pred_word_clusters = self._get_word_clusters(document, character_scores, fine_scores, top_indices)
        gold_character_heads = document.word_head_ids
        pred_character_heads = (character_scores > 0).to(torch.long).tolist()
        result.add_word_clusters(document, gold_word_clusters, pred_word_clusters)
        result.add_characters(document, gold_character_heads, pred_character_heads)

        # Span Prediction
        if self.model.training:
            heads, starts, ends = self._get_sp_training_data(document)
        else:
            heads = self._get_sp_inference_data(pred_word_clusters)

        sp_scores = []
        sp_dataloader = self._get_sp_dataloader(heads)
        for batch in sp_dataloader:
            scores = self.model.span_predictor(word_embeddings, *batch)
            sp_scores.append(scores)
        sp_scores = torch.cat(sp_scores, dim=0)

        if self.model.training:
            sp_loss = self.model.sp_loss(sp_scores, starts, ends, self.avg_n_train_heads)
        else:
            starts = sp_scores[:, :, 0].argmax(dim=1).tolist()
            ends = sp_scores[:, :, 1].argmax(dim=1).tolist()
            head2span = {head: (start, end) for head, start, end in zip(heads.tolist(), starts, ends)}
            pred_span_clusters = [set([head2span[head] for head in cluster]) for cluster in pred_word_clusters]
            self._remove_repeated_mentions(pred_span_clusters)
            gold_span_clusters = [set([(mention.begin, mention.end) for mention in cluster]) for cluster in document.clusters.values()]
            result.add_span_clusters(document, gold_span_clusters, pred_span_clusters)
            sp_loss = torch.zeros_like(coref_loss)

        return character_loss, coref_loss, sp_loss, result

    def _step(self):
        """Update model weights"""
        if not self.freeze_bert:
            self.bert_optimizer.step()
            if self.warmup_epochs >= 0:
                self.bert_scheduler.step()
        self.cr_optimizer.step()
        self.coref_optimizer.step()
        if self.warmup_epochs >= 0:
            self.cr_scheduler.step()
            self.coref_scheduler.step()

    def _train(self, document: CorefDocument) -> tuple[float, float, float]:
        """Train model on document and return character, coreference, and span prediction loss"""
        self.model.train()
        if not self.freeze_bert:
            self.bert_optimizer.zero_grad()
        self.cr_optimizer.zero_grad()
        self.coref_optimizer.zero_grad()
        character_loss, coref_loss, sp_loss, _ = self._run(document)
        (character_loss + coref_loss + sp_loss).backward()
        self._step()
        return character_loss.item(), coref_loss.item(), sp_loss.item()
    
    def _train_and_log(self):
        """Train and log"""
        inds = np.random.permutation(len(self.train_corpus))
        character_losses, coref_losses, sp_losses = [], [], []
        tbar = tqdm.tqdm(inds, unit="script")
        for doc_index in tbar:
            document = self.train_corpus[doc_index]
            character_loss, coref_loss, sp_loss = self._train(document)
            character_losses.append(character_loss)
            coref_losses.append(coref_loss)
            sp_losses.append(sp_loss)
            tbar.set_description(f"loss: character={np.mean(character_losses):.4f} coref={np.mean(coref_losses):.4f} sp={np.mean(sp_losses):.4f}")
    
    def _eval(self, corpus: CorefCorpus, name: str) -> tuple[CorefResult, float, float, float]:
        """Evaluate model on corpus and return metric results, average character loss, average coreference loss, and average conll coreference F1 (span)"""
        self.model.eval()
        with torch.no_grad():
            result = CorefResult(self.reference_scorer_file, self.output_dir, name)
            character_losses, coref_losses = [], []
            for document in corpus:
                _character_loss, _coref_loss, _, _result = self._run(document)
                character_losses.append(_character_loss.item())
                coref_losses.append(_coref_loss.item())
                document.predicted_word_clusters, document.predicted_span_clusters, document.predicted_heads = _result[document.movie]
                result.add(_result)
            return result, float(np.mean(character_losses)), float(np.mean(coref_losses)), float(result.span_score)
    
    def _eval_and_log(self) -> tuple[float, float, float, float]:
        """Evaluate and log, return character + coref loss for dev and train set, and average F1 score of dev and train set. If train set is not evaluated, train loss and score in -inf"""
        dev_result, dev_cr_loss, dev_coref_loss, dev_span_score = self._eval(self.dev_corpus, "dev")
        self._log(f"dev corpus: character_loss = {dev_cr_loss:.4f}, coref_loss = {dev_coref_loss:.4f}, {dev_result}")
        dev_loss = dev_cr_loss + dev_coref_loss
        if self.evaluate_train:
            train_result, train_cr_loss, train_coref_loss, train_span_score = self._eval(self.train_corpus, "train")
            self._log(f"train corpus: character_loss = {train_cr_loss:.4f}, coref_loss = {train_coref_loss:.4f}, {train_result}")
            train_loss = train_cr_loss + train_coref_loss
        else:
            train_span_score = -np.inf
            train_loss = -np.inf
        return dev_loss, train_loss, dev_span_score, train_span_score

    def _save(self, dev_loss: float, train_loss: float, dev_score: float, train_score: float):
        """Save model weights and predictions"""
        if self.save_model:
            model_file = os.path.join(self.output_dir, "movie_coref.pt")
            self.model.save_weights(model_file)
        if self.save_predictions:
            dev_file = os.path.join(self.output_dir, "dev.jsonlines")
            train_file = os.path.join(self.output_dir, "train.jsonlines")
            args = [[self.dev_corpus, dev_file], [self.train_corpus, train_file]] if self.evaluate_train else [[self.dev_corpus, dev_file]]
            for corpus, file in args:
                with jsonlines.open(file, "w") as writer:
                    for doc in corpus:
                        gold_clusters = [[[mention.begin, mention.end, mention.head] for mention in cluster] for cluster in doc.clusters.values()]
                        data = dict(movie=doc.movie, rater=doc.rater, token=doc.token, parse=doc.parse, pos=doc.pos, ner=doc.ner, is_pronoun=doc.is_pronoun, is_punctuation=doc.is_punctuation,
                                    speaker=doc.speaker, gold=gold_clusters, pred_word=doc.predicted_word_clusters, pred_span=doc.predicted_span_clusters, pred_head=doc.predicted_heads)
                        writer.write(data)
            for arg in ["gold", "pred"]:
                src = os.path.join(self.output_dir, f"{arg}_epoch.span.dev.conll")
                dst = os.path.join(self.output_dir, f"{arg}.span.dev.conll")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        if self.save_log:
            result_file = os.path.join(self.output_dir, "result.yaml")
            result = dict(character_recognition=dict(tag_embedding_size=self.tag_embedding_size, gru_nlayers=self.gru_nlayers, gru_hidden_size=self.gru_hidden_size, 
                                                     gru_bidirectional=self.gru_bidirectional),
                          topk=self.topk, dropout=self.dropout, freeze_bert=self.freeze_bert, genre=self.genre, bce_weight=self.bce_weight, bert_lr=self.bert_lr, character_lr=self.character_lr,
                          coref_lr=self.coref_lr, weight_decay=self.weight_decay, train_document_len=self.train_document_len, subword_batch_size=self.subword_batch_size, cr_seq_len=self.cr_seq_len,
                          cr_batch_size=self.cr_batch_size, fn_batch_size=self.fn_batch_size, sp_batch_size=self.sp_batch_size, add_cr_to_coarse=self.add_cr_to_coarse,
                          filter_mentions_by_cr=self.filter_mentions_by_cr, remove_singleton_cr=self.remove_singleton_cr, epoch=self.epoch, dev_loss=round(dev_loss, 4), dev_score=round(dev_score, 3),
                          train_loss=round(train_loss, 4), train_score=round(train_score, 3), preprocess=self.preprocess, warmup=self.warmup_epochs)
            with open(result_file, "w") as file:
                yaml.dump(result, file)

    def _save_plot(self, dev_losses: list[float], train_losses: list[float], dev_scores: list[float], train_scores: list[float]):
        """Plot loss and score curves"""
        if self.save_loss_curve:
            loss_file = os.path.join(self.output_dir, "loss.png")
            plt.subplot(1, 2, 1)
            if not np.all(np.array(train_losses) == -np.inf):
                plt.plot(np.arange(len(train_losses)) + 1, train_losses, label="train", lw=4, color="b")
            plt.plot(np.arange(len(dev_losses)) + 1, dev_losses, label="dev", lw=4, color="r")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.suptitle("Loss")
            plt.subplot(1, 2, 2)
            if not np.all(np.array(train_scores) == -np.inf):
                plt.plot(np.arange(len(train_scores)) + 1, train_scores, label="train", lw=4, color="b")
            plt.plot(np.arange(len(dev_scores)) + 1, dev_scores, label="dev", lw=4, color="r")
            plt.xlabel("epoch")
            plt.ylabel("avg F1")
            plt.legend()
            plt.suptitle("score")
            plt.savefig(loss_file)
            plt.close("all")
    
    def _clean(self):
        if self.output_dir is not None:
            for arg1 in ["word", "span"]:
                for arg2 in ["gold", "pred"]:
                    for arg3 in ["dev", "train"]:
                        pth = os.path.join(self.output_dir, f"{arg2}_epoch.{arg1}.{arg3}.conll")
                        if os.path.exists(pth):
                            os.remove(pth)

    def __call__(self):
        self._log("")
        self._log(self._gpu_usage())
        max_left, max_right = -1, -1

        # Create model
        self._log("\nInitializing model")
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

        # Load model weights
        self._log("\nLoading model weights from word-level-coref model")
        self.model.load_weights(self.weights_file)
        self.model.device = self.device

        # Load training set
        self._log("Loading training corpus")
        self.train_corpus = CorefCorpus(self.train_file)
        self.train_corpus = self._prepare_corpus(self.train_corpus, self.train_document_len, 0, exclude_subdocuments_with_no_clusters=True)

        # Setting some model hyperparameters based on training set
        self.avg_n_train_heads = self._find_avg_n_heads(self.train_corpus)
        max_left, max_right = self._find_max_n_words_left_and_right_of_head(self.train_corpus)
        self.model.span_predictor.max_left = max_left
        self.model.span_predictor.max_right = max_right

        # Load development set
        self._log("Loading development corpus")
        self.dev_corpus = CorefCorpus(self.dev_file)
        self.dev_corpus = self._prepare_corpus(self.dev_corpus)
        self._log("\n")

        # Create optimizers
        self._create_optimizers()

        # Sanity check for eval
        self.epoch = 0
        if self.epoch > self.n_epochs_no_eval:
            self._eval_and_log()
            self._log("\n")

        # Training loop
        max_dev_score = -np.inf
        n_epochs_early_stopping = 0
        dev_losses, train_losses, dev_scores, train_scores = [], [], [], []
        self._log("Starting training\n")
        for self.epoch in range(1, self.max_epochs + 1):
            self._log(f"Epoch = {self.epoch}")
            self._train_and_log()
            if self.epoch > self.n_epochs_no_eval:
                dev_loss, train_loss, dev_score, train_score = self._eval_and_log()
                dev_losses.append(dev_loss)
                train_losses.append(train_loss)
                dev_scores.append(dev_score)
                train_scores.append(train_score)
                if dev_score > max_dev_score:
                    max_dev_score = dev_score
                    n_epochs_early_stopping = 0
                    self._save(dev_loss, train_loss, dev_score, train_score)
                else:
                    n_epochs_early_stopping += 1
                self._log(f"Dev score decreasing for {n_epochs_early_stopping} epochs, {self.patience - n_epochs_early_stopping} epochs left till early stop")
                if n_epochs_early_stopping == self.patience:
                    break
            self._log("\n")
        
        # Loss Curve
        self._save_plot(dev_losses, train_losses, dev_scores, train_scores)

        # Clean
        self._clean()