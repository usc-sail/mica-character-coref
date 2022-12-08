"""Train movie coreference model on single GPU"""
from mica_text_coref.coref.movie_coref.coreference import model
from mica_text_coref.coref.movie_coref import data
from mica_text_coref.coref.movie_coref import split_and_merge
from mica_text_coref.coref.movie_coref import evaluate

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
import copy
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

EvalOutput = tuple[float, data.MovieCorefMetric, list[data.CorefResult]]
StrategyEvalOutput = list[tuple[float, data.MovieCorefMetric, data.CorefResult]]

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
        test_movie: str = None,
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
        test_document_lens: list[int] = [1024, 2048, 3072, 4096, 5120],
        test_overlap_lens: list[int] = [0, 128, 256, 384, 512],
        test_merge_strategies: list[str] = ["none", "pre", "post", "max", "min", "avg"],
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
            test_movie: train set movie left-out for cross validation, if none entire train set is used
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
            test_document_lens: size of the test subdocuments in words
            test_overlap_lens: size of overlap between adjacent test subdocuments in words
            test_merge_strategies: strategy to merge coref scores of adjacent test subdocuments
            subword_batch_size: number of subword sequences in batch
            cr_seq_len: sequence length for character recognizer model
            cr_batch_size: number of sequences in a character recognition batch
            fn_batch_size: number of word pairs in fine scoring batch
            sp_batch_size: number of heads in span prediction batch
            evaluate_train: if true, evaluate training set as well
            n_epochs_no_eval: initial number of epochs for which evaluation is not done, set it greater than zero if
                the model is not stable in the initial few epochs
            add_cr_to_coarse: if true, add character head scores to coarse scores
            filter_mentions_by_cr: if true, only keep words whose character score is positive while constructing the
                word clusters during inference
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
        self.test_movie = test_movie
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
        self.test_document_lens = test_document_lens
        self.test_overlap_lens = test_overlap_lens
        self.test_merge_strategies = test_merge_strategies
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
            self._log(f"{arg:25s} = {value}")

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
                self.bert_scheduler = get_linear_schedule_with_warmup(
                    self.bert_optimizer, self.warmup_epochs * len(self.train_corpus),
                    len(self.train_corpus) * self.max_epochs)
        self.cr_optimizer = AdamW(self.model.cr_parameters(), lr=self.character_lr, weight_decay=self.weight_decay)
        self.coref_optimizer = AdamW(self.model.coref_parameters(), lr=self.coref_lr, weight_decay=self.weight_decay)
        if self.warmup_epochs >= 0:
            self.cr_scheduler = get_linear_schedule_with_warmup(
                self.cr_optimizer, self.warmup_epochs * len(self.train_corpus),
                len(self.train_corpus) * self.max_epochs)
            self.coref_scheduler = get_linear_schedule_with_warmup(
                self.coref_optimizer, self.warmup_epochs * len(self.train_corpus),
                len(self.train_corpus) * self.max_epochs)

    def _split_screenplay(self, document: data.CorefDocument, split_len: int, overlap_len: int, 
                          exclude_subdocuments_with_no_clusters: bool = True, verbose = False):
        """Split screenplay document into smaller documents

        Args:
            document: CorefDocument object representing the original screenplay document
            split_len: Length of the smaller CorefDocument objects in words
            overlap_len: number of words overlapping between successive smaller CorefDocuments
            exclude_subdocuments_with_no_clusters: if true, exclude subdocuments if they contain no clusters
        
        Returns:
            Generator of CorefDocument objects.
        """
        # initialize offsets and sentence offsets
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
            _document = data.CorefDocument()

            # populate subdocument-length fields
            _document.movie = document.movie + f"_{k + 1}"
            _document.rater = document.rater
            _document.token = document.token[i: j]
            _document.parse = document.parse[i: j]
            _document.parse_ids = [data.parse_labelset[x] for x in _document.parse]
            _document.pos = document.pos[i: j]
            _document.pos_ids = [data.pos_labelset[x] for x in _document.pos]
            _document.ner = document.ner[i: j]
            _document.ner_ids = [data.ner_labelset[x] for x in _document.ner]
            _document.is_pronoun = document.is_pronoun[i: j]
            _document.is_punctuation = document.is_punctuation[i: j]
            _document.speaker = document.speaker[i: j]

            # populate sentence offsets
            si = np.nonzero(sentence_offsets[:,0] == i)[0][0]
            sj = np.nonzero(sentence_offsets[:,1] == j - 1)[0][0] + 1
            _document.sentence_offsets = (sentence_offsets[si: sj] - sentence_offsets[si, 0]).tolist()

            # populate clusters
            clusters: dict[str, set[data.Mention]] = collections.defaultdict(set)
            n_mentions = 0
            for character, mentions in document.clusters.items():
                for mention in mentions:
                    assert (mention.end < i or i <= mention.begin <= mention.end < j or j <= mention.begin), (
                        f"Mention crosses subdocument boundaries mention={mention} i={i} j={j}")
                    if i <= mention.begin <= mention.end < j:
                        new_mention = data.Mention(mention.begin - i, mention.end - i, mention.head - i)
                        clusters[character].add(new_mention)
                        n_mentions += 1
            
            # go to next document if clusters is empty
            if exclude_subdocuments_with_no_clusters and len(clusters) == 0:
                continue

            # fill the clusters field and its derivaties, and the offset
            _document.clusters = clusters
            _document.word_cluster_ids = document.word_cluster_ids[i: j]
            _document.word_head_ids = document.word_head_ids[i: j]
            _document.offset = (i, j)
            if verbose:
                self._log(f"{_document.movie}: {len(_document.token)} words, {n_mentions} mentions, "
                          f"{len(_document.clusters)} clusters")
            yield _document

    def _tokenize_document(self, document: data.CorefDocument, verbose = False):
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

    def _create_subword_dataloader(self, document: data.CorefDocument, verbose = False):
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
        document.subword_dataloader = DataLoader(TensorDataset(
            subword_id_tensor, subword_mask_tensor), batch_size=self.subword_batch_size)
        if verbose:
            self._log(f"{document.movie}: {len(subword_id_seqs)} subword sequences")

    def _prepare_corpus(self, corpus: data.CorefCorpus, split_len: int = None, overlap_len = 0, 
                        exclude_subdocuments_with_no_clusters: bool = True, verbose = False) -> data.CorefCorpus:
        """Create new corpus by splitting, tokenizing, and creating subword sequences
        of the screenplay documents in the original corpus

        Args:
            corpus: CorefCorpus containing screenplay documents
            split_len: length of the new subdocuments in words the screenplay documents will be split into,
                if split_len is None then no splitting occurs
            overlap_len: number of words overlapping between successive subdocuments, ignored if no splitting occurs
            exclude_subdocuments_with_no_clusters: if true, exclude subdocuments if they contain no clusters
        
        Returns:
            CorefCorpus: The modified CorefCorpus
        """
        _corpus = data.CorefCorpus()
        if split_len is None:
            document_generator = corpus
        else:
            document_generator = itertools.chain(*[self._split_screenplay(
                document, split_len, overlap_len,
                exclude_subdocuments_with_no_clusters=exclude_subdocuments_with_no_clusters, 
                verbose=verbose) for document in corpus])
        for _document in document_generator:
            self._tokenize_document(_document, verbose=verbose)
            self._create_subword_dataloader(_document, verbose=verbose)
            _corpus.documents.append(_document)
        return _corpus

    def _find_max_n_words_left_and_right_of_head(self, corpus: data.CorefCorpus) -> tuple[int, int]:
        """Get the maximum number of words to the left and right of a head word"""
        max_left, max_right = -1, -1
        for document in corpus:
            for mentions in document.clusters.values():
                for mention in mentions:
                    max_left = max(max_left, mention.head - mention.begin)
                    max_right = max(max_right, mention.end - mention.head)
        return max_left, max_right
    
    def _find_avg_n_heads(self, corpus: data.CorefCorpus) -> float:
        """Get average number of character mention heads per document."""
        n_heads = []
        for document in corpus:
            n_document_heads = 0
            for mentions in document.clusters.values():
                n_document_heads += len(mentions)
            n_heads.append(n_document_heads)
        return np.mean(n_heads)

    def _create_cr_dataloader(self, word_embeddings: torch.FloatTensor, document: data.CorefDocument,
                              verbose = False) -> DataLoader:
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
                _seq_parsetag_ids += [data.parse_labelset.other_id] * padding_size
                _seq_postag_ids += [data.pos_labelset.other_id] * padding_size
                _seq_nertag_ids += [data.ner_labelset.other_id] * padding_size
                _seq_ispronoun += [False] * padding_size
                _seq_ispunctuation += [False] * padding_size
            seq_parsetag_ids.append(_seq_parsetag_ids)
            seq_postag_ids.append(_seq_postag_ids)
            seq_nertag_ids.append(_seq_nertag_ids)
            seq_ispronoun.append(_seq_ispronoun)
            seq_ispunctuation.append(_seq_ispunctuation)
        
        # tensorize sequences
        tensor_word_embeddings = torch.cat([
            word_embeddings, 
            torch.zeros((n_seqs * self.cr_seq_len - len(word_embeddings), word_embeddings.shape[1]), 
                        device=self.device)
            ], dim=0).view((n_seqs, self.cr_seq_len, -1))
        tensor_parsetag_ids = torch.LongTensor(seq_parsetag_ids).to(self.device)
        tensor_postag_ids = torch.LongTensor(seq_postag_ids).to(self.device)
        tensor_nertag_ids = torch.LongTensor(seq_nertag_ids).to(self.device)
        tensor_ispronoun = torch.LongTensor(seq_ispronoun).to(self.device)
        tensor_ispunctuation = torch.LongTensor(seq_parsetag_ids).to(self.device)
        dataloader = DataLoader(TensorDataset(tensor_word_embeddings, tensor_parsetag_ids, tensor_postag_ids,
                                              tensor_nertag_ids, tensor_ispronoun, tensor_ispunctuation),
                                              batch_size=self.cr_batch_size)
        batch_size = math.ceil(len(tensor_word_embeddings)/len(dataloader))
        if verbose:
            prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
            self._log(f"{prefix}: cr batch size = {batch_size} sequences/batch")
        return dataloader

    def _create_fine_dataloader(self, word_embeddings: torch.FloatTensor, features: torch.FloatTensor,
                                scores: torch.FloatTensor, indices: torch.LongTensor) -> DataLoader:
        """Get dataloader for fine coreference scoring

        Args:
            word_embeddings: Float tensor of word embeddings [n_words, embedding_size]
            features: Float tensor of pairwise features (genre, same speaker, and distance) of words
                and its top k scoring antecedents. [n_words, n_antecedents, feature_size]
            scores: Float tensor of coarse coreference scores of words and its top k scoring antecedents.
                [n_words, n_antecedents]
            indices: Long tensor of the word indices of the top scoring antecedents. [n_words, n_antecedents].
                Lies between 0 and n_words - 1
        
        Returns:
            Torch dataloader.
        """
        dataset = TensorDataset(word_embeddings, features, indices, scores)
        dataloader = DataLoader(dataset, batch_size=self.fn_batch_size)
        return dataloader

    def _get_coref_ground_truth(self, cluster_ids: list[int], top_indices: torch.LongTensor,
                                valid_pair_map: torch.BoolTensor) -> torch.FloatTensor:
        """Get coreference ground truth for evaluation

        Args:
            cluster_ids: List of word cluster ids. 0 if word is not a character head, >=1 if word is a character head.
                Co-referring words have same cluster id. [n_words]
            top_indices: Long tensor of top scoring antecedents. [n_words, n_antecedents]
            valid_pair_mask: Bool tensor of whether the word-antecedent pair is valid (word comes after antecedents).
                [n_words, n_antecedents]
        
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

    def _clusterize(self, character_scores: torch.FloatTensor, coref_scores: torch.FloatTensor,
                    indices: torch.LongTensor) -> list[set[int]]:
        """Find word-level clusters from character head scores, coreference scores, and the top indices.

        Args:
            character_scores: Float tensor of word-level logits of the word being the head of a character mention.
                [n_words]
            coref_scores: Float tensor of logits of the word pair being coreferent with each other.
                The first column contains all zeros. [n_words, 1 + n_antecedents]
            indices: Long tensor of antecedent indices. [n_words, n_antecedents]
        
        Returns:
            Word clusters: list[set[int]]
        """
        # find antecedents of words
        is_character = (character_scores > 0).tolist()
        antecedents = coref_scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(coref_scores))[not_dummy]
        antecedents = indices[coref_span_heads, antecedents[not_dummy]]

        # link word with antecedents in a graph
        nodes = [data.GraphNode(i) for i in range(len(coref_scores))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            if not self.filter_mentions_by_cr or (is_character[i] and is_character[j]):
                nodes[i].link(nodes[j])
                assert nodes[i] is not nodes[j]

        # find components in the graph and get the predicted clusters
        clusters = []
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
                    clusters.append(cluster)

        # check mentions don't repeat between clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                assert clusters[i].isdisjoint(clusters[j]), (
                    f"mentions repeat in predicted cluster {clusters[i]} and cluster {clusters[j]}")

        return clusters

    def _get_sp_training_data(self, document: data.CorefDocument) -> tuple[torch.LongTensor, torch.LongTensor,
                                                                           torch.LongTensor]:
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

    def _run(self, document: data.CorefDocument) -> data.CorefResult:
        """Model coreference in document."""
        result = data.CorefResult()

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
        fn_dataloader = self._create_fine_dataloader(word_embeddings, features, coarse_scores, top_indices)
        for batch in fn_dataloader:
            scores = self.model.fine_scorer(word_embeddings, *batch)
            fine_scores.append(scores)
        fine_scores = torch.cat(fine_scores, dim=0)

        # Compute loss
        coref_y = self._get_coref_ground_truth(document.word_cluster_ids, top_indices, (coarse_scores > float("-inf")))
        coref_loss = self.model.coref_loss(fine_scores, coref_y)
        character_y = torch.FloatTensor(document.word_head_ids).to(self.device)
        character_loss = self.model.cr_loss(character_scores, character_y)

        # Predict character heads and word clusters
        word_clusters = self._clusterize(character_scores, fine_scores, top_indices)
        character_heads = (character_scores > 0).to(torch.long).cpu().numpy()

        # Fill result
        result.character_scores = character_scores
        result.top_indices = top_indices
        result.coref_scores = fine_scores
        result.coref_loss = coref_loss
        result.character_loss = character_loss
        result.predicted_character_heads = character_heads
        result.predicted_word_clusters = word_clusters

        # Span prediction
        if self.model.training:
            heads, starts, ends = self._get_sp_training_data(document)
        else:
            heads = self._get_sp_inference_data(word_clusters)
        sp_scores = []
        sp_dataloader = self._get_sp_dataloader(heads)
        for batch in sp_dataloader:
            scores = self.model.span_predictor(word_embeddings, *batch)
            sp_scores.append(scores)
        if sp_scores:
            sp_scores = torch.cat(sp_scores, dim=0)
        if self.model.training:
            span_loss = self.model.sp_loss(sp_scores, starts, ends, self.avg_n_train_heads)
            result.span_loss = span_loss
        elif len(sp_scores) > 0:
            starts = sp_scores[:, :, 0].argmax(dim=1).tolist()
            ends = sp_scores[:, :, 1].argmax(dim=1).tolist()
            max_sp_scores = (sp_scores[:, :, 0].max(dim=1)[0] + sp_scores[:, :, 1].max(dim=1)[0]).tolist()
            head2span = {head: (start, end, score) for head, start, end, score in zip(heads.tolist(), starts, ends,
                                                                                      max_sp_scores)}
            span_clusters = [set([head2span[head][:2] for head in cluster]) for cluster in word_clusters]
            self._remove_repeated_mentions(span_clusters)
            result.head2span = head2span
            result.predicted_span_clusters = span_clusters

        return result

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

    def _train_document(self, document: data.CorefDocument) -> data.CorefResult:
        """Train model on document and return CorefResult object"""
        self.model.train()
        if not self.freeze_bert:
            self.bert_optimizer.zero_grad()
        self.cr_optimizer.zero_grad()
        self.coref_optimizer.zero_grad()
        result = self._run(document)
        loss = result.character_loss + result.coref_loss + result.span_loss
        loss.backward()
        self._step()
        return result
    
    def _train(self):
        """Train and log"""
        inds = np.random.permutation(len(self.train_corpus))
        character_losses, coref_losses, sp_losses = [], [], []
        tbar = tqdm.tqdm(inds, unit="script")
        for doc_index in tbar:
            document = self.train_corpus[doc_index]
            result = self._train_document(document)
            character_losses.append(result.character_loss.item())
            coref_losses.append(result.coref_loss.item())
            sp_losses.append(result.span_loss.item())
            tbar.set_description(f"{document.movie:25s}:: character={np.mean(character_losses):.4f} "
                                 f"coref={np.mean(coref_losses):.4f} span={np.mean(sp_losses):.4f}")

    def _merge(self, documents: list[data.CorefDocument], results: list[data.CorefResult],
               strategy: str) -> data.CorefResult:
        """Merge the CorefResult of the subdocuments

        Args:
            documents: List of CorefDocuments.
            results: List of CorefResults.
            strategy: Merge strategy.
        
        Returns:
            Merged CorefResult
        """
        character_scores_arr, coref_scores_arr, inds_arr, head2spans, overlap_lens = [], [], [], [], []
        for i in range(len(documents)):
            offset = documents[i].offset
            ind = results[i].top_indices + offset[0]
            coref_scores = results[i].coref_scores
            if ind.shape[1] < self.topk:
                d = self.topk - ind.shape[1]
                ind = torch.cat((ind, ind[:,:d]), dim=1)
                coref_scores = torch.cat((coref_scores, coref_scores[:,1:d+1]), dim=1)
            character_scores_arr.append(results[i].character_scores)
            coref_scores_arr.append(coref_scores)
            inds_arr.append(ind)
            head2span = {}
            for head, (start, end, score) in results[i].head2span.items():
                head2span[head + offset[0]] = (start + offset[0], end + offset[0], score)
            head2spans.append(head2span)
            if i > 0:
                overlap_lens.append(documents[i - 1].offset[1] - documents[i].offset[0])
        merged_coref_scores, merged_inds = split_and_merge.combine_coref_scores(coref_scores_arr, inds_arr, 
                                                                                overlap_lens, strategy)
        merged_character_scores = split_and_merge.combine_character_scores(character_scores_arr, overlap_lens,
                                                                           strategy)
        merged_head2span = split_and_merge.combine_head2spans(head2spans)
        merged_result = data.CorefResult()
        merged_result.character_loss = torch.mean(torch.tensor([result.character_loss for result in results]))
        merged_result.coref_loss = torch.mean(torch.tensor([result.coref_loss for result in results]))
        merged_result.character_scores = merged_character_scores
        merged_result.coref_scores = merged_coref_scores
        merged_result.top_indices = merged_inds
        merged_result.head2span = merged_head2span
        merged_result.predicted_character_heads = (merged_character_scores > 0).to(torch.long).cpu().numpy()
        merged_result.predicted_word_clusters = self._clusterize(merged_character_scores, merged_coref_scores,
                                                                 merged_inds)
        span_clusters = []
        for cluster in merged_result.predicted_word_clusters:
            span_cluster = set(merged_head2span[head][:2] for head in cluster if head in merged_head2span)
            if span_cluster:
                span_clusters.append(span_cluster)
        merged_result.predicted_span_clusters = span_clusters
        return merged_result

    def _eval_subdocuments_and_merge(self, document: data.CorefDocument, subdocument_corpus: data.CorefCorpus,
                                     strategies: list[str], name: str) -> StrategyEvalOutput:
        """Run model inference on the subdocuments of the corpus, merge results according to strategy, and evaluate
        output.

        Args:
            document: CorefDocument which was split to obtain the subdocuments.
            subdocument_corpus: CorefCorpus containing the subdocuments.
            strategies: Merge strategies.
            name: "test_<document_len>_<overlap_len>" filename to be included in CONLL filenames.

        Returns:
            A list of tuples, each tuple corresponding to a strategy and containing the following:
                loss: character loss + coreference loss
                metric: MovieCorefMetric
                results: CorefResult
        """
        torch.cuda.empty_cache()
        self.model.eval()
        results = []
        with torch.no_grad():
            for subdocument in subdocument_corpus:
                result = self._run(subdocument)
                results.append(result)
        strategy_tuples = []
        tbar = tqdm.tqdm(strategies, unit="strategy")
        for strategy in tbar:
            tbar.set_description(strategy)
            strategy_result = self._merge(subdocument_corpus.documents, results, strategy)
            strategy_metric = evaluate.evaluate([document], [strategy_result], self.reference_scorer_file,
                                                self.output_dir, f"{name}_{strategy}")
            loss = (strategy_result.character_loss + strategy_result.coref_loss).item()
            strategy_tuples.append((loss, strategy_metric, strategy_result))
        return strategy_tuples

    def _test(self) -> dict[tuple[int, int], StrategyEvalOutput]:
        """Test model on left-out-movie for each subdocument length, overlap length, and strategy"""
        test_strategy_eval_output: dict[tuple[int, int], StrategyEvalOutput] = {}
        tbar = tqdm.tqdm(self.test_document_and_overlap_len_to_corpus.items())
        for (document_len, overlap_len), subdocument_corpus in tbar:
            tbar.set_description(f"subdocument={document_len:<5d} overlap={overlap_len:<4d}::")
            strategy_eval_output = self._eval_subdocuments_and_merge(
                self.test_document, subdocument_corpus, self.test_merge_strategies,
                f"test_{document_len}_{overlap_len}")
            test_strategy_eval_output[(document_len, overlap_len)] = strategy_eval_output
        for (document_len, overlap_len), eval_outputs in test_strategy_eval_output.items():
            for strategy, eval_output in zip(self.test_merge_strategies, eval_outputs):
                self._log(f"test:: subdocument={document_len:5d} overlap={overlap_len:4d} strategy={strategy:4s} "
                            f"loss={eval_output[0]}, metric:{eval_output[1]}")
        return test_strategy_eval_output
    
    def _eval_corpus(self, corpus: data.CorefCorpus, name: str) -> EvalOutput:
        """Run model inference on corpus and evaluate output

        Args:
            corpus: CorefCorpus to evaluate
            name: "train" or "dev"
        
        Returns:
            loss: character loss + coreference loss
            metric: MovieCorefMetric
            results: List of CorefResult for each CorefDocument
        """
        torch.cuda.empty_cache()
        self.model.eval()
        results = []
        with torch.no_grad():
            for document in corpus:
                result = self._run(document)
                results.append(result)
        movie_coref_metric = evaluate.evaluate(corpus.documents, results, self.reference_scorer_file,
                                               self.output_dir, name)
        character_loss = float(np.mean([result.character_loss.item() for result in results]))
        coref_loss = float(np.mean([result.coref_loss.item() for result in results]))
        return character_loss + coref_loss, movie_coref_metric, results

    def _eval_dev(self) -> EvalOutput:
        """Evaluate dev corpus and return evaluation output"""
        dev_eval_output = self._eval_corpus(self.dev_corpus, "dev")
        self._log(f"dev:: loss={dev_eval_output[0]:.4f}, metric:{dev_eval_output[1]}")
        return dev_eval_output
    
    def _eval_train(self) -> EvalOutput:
        """Evaluate train corpus and return evaluation output"""
        train_eval_output = self._eval_corpus(self.train_corpus, "train")
        self._log(f"train:: loss={train_eval_output[0]:.4f}, metric:{train_eval_output[1]}")
        return train_eval_output

    def _save_predictions(self, corpus: data.CorefCorpus, eval_output: EvalOutput, name: str):
        """Save predictions"""
        file = os.path.join(self.output_dir, f"{name}.jsonlines")
        with jsonlines.open(file, "w") as writer:
            for doc, result in zip(corpus, eval_output[2]):
                gold_clusters = [[[mention.begin, mention.end, mention.head] for mention in cluster]
                                        for cluster in doc.clusters.values()]
                pred_word_clusters = [sorted(word_cluster) for word_cluster in result.predicted_word_clusters]
                pred_span_clusters = [sorted([list(span) for span in span_cluster])
                                            for span_cluster in result.predicted_span_clusters]
                pred_heads = result.predicted_character_heads.tolist()
                data = dict(movie=doc.movie,
                            rater=doc.rater,
                            token=doc.token,
                            parse=doc.parse,
                            pos=doc.pos,
                            ner=doc.ner,
                            is_pronoun=doc.is_pronoun,
                            is_punctuation=doc.is_punctuation,
                            speaker=doc.speaker,
                            gold=gold_clusters,
                            pred_word=pred_word_clusters,
                            pred_span=pred_span_clusters,
                            pred_head=pred_heads)
                writer.write(data)

    def _save_test_predictions(self):
        """Save predictions of the test set"""
        for (document_len, overlap_len), strategy_eval_output in (
                self.test_document_and_overlap_len_to_eval_outputs.items()):
            for strategy, eval_output in zip(self.test_merge_strategies, strategy_eval_output):
                test_file = os.path.join(self.output_dir,
                                            f"test_{document_len}_{overlap_len}_{strategy}.jsonlines")
                with jsonlines.open(test_file, "w") as writer:
                    gold_clusters = [[[mention.begin, mention.end, mention.head] for mention in cluster]
                                            for cluster in self.test_document.clusters.values()]
                    pred_word_clusters = [sorted(word_cluster)
                                                for word_cluster in eval_output[2].predicted_word_clusters]
                    pred_span_clusters = [sorted([list(span) for span in span_cluster])
                                                for span_cluster in eval_output[2].predicted_span_clusters]
                    pred_heads = eval_output[2].predicted_character_heads.tolist()
                    data = dict(movie=self.test_document.movie,
                                rater=self.test_document.rater,
                                token=self.test_document.token,
                                parse=self.test_document.parse,
                                pos=self.test_document.pos,
                                ner=self.test_document.ner,
                                is_pronoun=self.test_document.is_pronoun,
                                is_punctuation=self.test_document.is_punctuation,
                                speaker=self.test_document.speaker,
                                gold=gold_clusters,
                                pred_word=pred_word_clusters,
                                pred_span=pred_span_clusters,
                                pred_head=pred_heads)
                    writer.write(data)
        
        # Copy test CONLL files for gold and pred span clusters
        for arg in ["gold", "pred"]:
            for document_len in self.test_document_lens:
                for overlap_len in self.test_overlap_lens:
                    for strategy in self.test_merge_strategies:
                        name = f"test_{document_len}_{overlap_len}_{strategy}"
                        src = os.path.join(self.output_dir, f"{arg}_epoch.span.{name}.conll")
                        dst = os.path.join(self.output_dir, f"{name}.{arg}.conll")
                        if os.path.exists(src):
                            shutil.copy2(src, dst)

    def _save_config(self):
        """Save hyperparams config and metric scores"""
        result_file = os.path.join(self.output_dir, "result.yaml")
        train_metric_dict, test_metric_dict = {}, {}
        if self.evaluate_train:
            train_metric_dict = self.train_eval_output[1].todict()
        if self.test_document_and_overlap_len_to_eval_outputs is not None:
            for (document_len, overlap_len), eval_outputs in (
                    self.test_document_and_overlap_len_to_eval_outputs.items()):
                if document_len not in test_metric_dict:
                    test_metric_dict[document_len] = {}
                if overlap_len not in test_metric_dict[document_len]:
                    test_metric_dict[document_len][overlap_len] = {}
                for strategy, eval_output in zip(self.test_merge_strategies, eval_outputs):
                    test_metric_dict[document_len][overlap_len][strategy] = eval_output[1].todict()
        result = dict(character_recognition=dict(tag_embedding_size=self.tag_embedding_size,
                                                 gru_nlayers=self.gru_nlayers,
                                                 gru_hidden_size=self.gru_hidden_size, 
                                                 gru_bidirectional=self.gru_bidirectional),
                      preprocess=self.preprocess,
                      test_movie=self.test_movie,
                      topk=self.topk,
                      dropout=self.dropout,
                      freeze_bert=self.freeze_bert,
                      genre=self.genre,
                      bce_weight=self.bce_weight,
                      bert_lr=self.bert_lr,
                      character_lr=self.character_lr,
                      coref_lr=self.coref_lr,
                      warmup=self.warmup_epochs,
                      n_steps_per_epoch=len(self.train_corpus),
                      weight_decay=self.weight_decay,
                      train_document_len=self.train_document_len,
                      subword_batch_size=self.subword_batch_size,
                      cr_seq_len=self.cr_seq_len,
                      cr_batch_size=self.cr_batch_size,
                      fn_batch_size=self.fn_batch_size,
                      sp_batch_size=self.sp_batch_size,
                      add_cr_to_coarse=self.add_cr_to_coarse,
                      filter_mentions_by_cr=self.filter_mentions_by_cr,
                      remove_singleton_cr=self.remove_singleton_cr,
                      epoch=self.epoch,
                      max_epochs=self.max_epochs,
                      n_epochs_no_eval=self.n_epochs_no_eval,
                      dev_losses=np.round(self.dev_losses, 4).tolist(),
                      train_losses=np.round(self.train_losses, 4).tolist(),
                      dev_scores=np.round(self.dev_scores, 4).tolist(),
                      train_scores=np.round(self.train_scores, 4).tolist(),
                      dev_metric=self.dev_eval_output[1].todict(),
                      train_metric=train_metric_dict,
                      test_metric=test_metric_dict
                      )
        with open(result_file, "w") as file:
            yaml.dump(result, file)

    def _save(self):
        """Save model weights and predictions"""
        # Save model weights
        if self.save_model:
            model_file = os.path.join(self.output_dir, "movie_coref.pt")
            self.model.save_weights(model_file)

        # Save predictions
        if self.save_predictions:
            self._save_predictions(self.dev_corpus, self.dev_eval_output, "dev")
            if self.evaluate_train:
                self._save_predictions(self.train_corpus, self.train_eval_output, "train")

            # Copy dev CONLL files for gold and pred span clusters
            for arg in ["gold", "pred"]:
                src = os.path.join(self.output_dir, f"{arg}_epoch.span.dev.conll")
                dst = os.path.join(self.output_dir, f"dev.{arg}.conll")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # Save hyperparams used and best metric
        if self.save_log:
            self._save_config()

    def _save_plot(self):
        """Plot loss and score curves"""
        if self.save_loss_curve:
            loss_file = os.path.join(self.output_dir, "loss.png")
            plt.figure(figsize=(8, 6))
            plt.subplot(1, 2, 1)
            if self.train_losses:
                plt.plot(np.arange(len(self.train_losses)) + 1, self.train_losses, label="train", lw=4, color="b")
            plt.plot(np.arange(len(self.dev_losses)) + 1, self.dev_losses, label="dev", lw=4, color="r")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.title("Loss")
            plt.grid(visible=True, which="both", axis="both")
            plt.subplot(1, 2, 2)
            if self.train_scores:
                plt.plot(np.arange(len(self.train_scores)) + 1, self.train_scores, label="train", lw=4, color="b")
            plt.plot(np.arange(len(self.dev_scores)) + 1, self.dev_scores, label="dev", lw=4, color="r")
            plt.xlabel("epoch")
            plt.ylabel("avg F1")
            plt.legend()
            plt.title("score")
            plt.grid(visible=True, which="both", axis="both")
            plt.savefig(loss_file)
            plt.close("all")
    
    def _clean(self):
        if self.output_dir is not None:
            for arg1 in ["word", "span"]:
                for arg2 in ["gold", "pred"]:
                    for arg3 in ["dev", "train"]:
                        path = os.path.join(self.output_dir, f"{arg2}_epoch.{arg1}.{arg3}.conll")
                        if os.path.exists(path):
                            os.remove(path)
                    for document_len in self.test_document_lens:
                        for overlap_len in self.test_overlap_lens:
                            for strategy in self.test_merge_strategies:
                                path = os.path.join(
                                    self.output_dir,
                                    f"{arg2}_epoch.{arg1}.test_{document_len}_{overlap_len}_{strategy}.conll")
                                if os.path.exists(path):
                                    os.remove(path)

    def __call__(self):
        self._log("")
        self._log(self._gpu_usage())
        max_left, max_right = -1, -1

        # Create model
        self._log("\nInitializing model")
        self.model = model.MovieCoreference(
            parsetag_size = len(data.parse_labelset),
            postag_size = len(data.pos_labelset),
            nertag_size = len(data.ner_labelset),
            tag_embedding_size = self.tag_embedding_size,
            gru_nlayers = self.gru_nlayers,
            gru_hidden_size = self.gru_hidden_size,
            gru_bidirectional = self.gru_bidirectional,
            topk = self.topk,
            bce_weight = self.bce_weight,
            dropout = self.dropout)

        # Load model weights
        self._log("\nLoading model weights from word-level-coref model")
        self.model.load_weights_from_file(self.weights_file)
        self.model.device = self.device

        # Load training set
        self._log("Loading training corpus")
        self.train_corpus = data.CorefCorpus(self.train_file)
        self.test = self.test_movie is not None and any(doc.movie == self.test_movie for doc in self.train_corpus)
        if self.test:
            self.test_document = [document for document in self.train_corpus if document.movie == self.test_movie][0]
            self.train_corpus.documents = [document for document in self.train_corpus 
                                                        if document.movie != self.test_movie]
        self.train_corpus = self._prepare_corpus(self.train_corpus, self.train_document_len, 0, 
                                                 exclude_subdocuments_with_no_clusters=True)

        # Setting some model hyperparameters based on training set
        self.avg_n_train_heads = self._find_avg_n_heads(self.train_corpus)
        max_left, max_right = self._find_max_n_words_left_and_right_of_head(self.train_corpus)
        self.model.span_predictor.max_left = max_left
        self.model.span_predictor.max_right = max_right

        # Load development set
        self._log("Loading development corpus")
        self.dev_corpus = data.CorefCorpus(self.dev_file)
        self.dev_corpus = self._prepare_corpus(self.dev_corpus)

        # Load testing set
        if self.test:
            self._log("Loading testing corpus")
            test_corpus = data.CorefCorpus()
            test_corpus.documents = [self.test_document]
            self.test_document_and_overlap_len_to_corpus: dict[tuple[int, int]: data.CorefCorpus] = {}
            for document_len in self.test_document_lens:
                for overlap_len in self.test_overlap_lens:
                    corpus = self._prepare_corpus(test_corpus, document_len, overlap_len, 
                                                  exclude_subdocuments_with_no_clusters=False)
                    self.test_document_and_overlap_len_to_corpus[(document_len, overlap_len)] = corpus
        self._log("\n")

        # Create optimizers
        self._create_optimizers()

        # Evaluation and Early-stopping variables
        max_dev_score = -np.inf
        n_epochs_early_stopping = 0
        self.dev_losses, self.train_losses, self.dev_scores, self.train_scores = [], [], [], []
        self.dev_eval_output: EvalOutput = None
        self.train_eval_output: EvalOutput = None
        self.test_document_and_overlap_len_to_eval_outputs: dict[tuple[int, int], StrategyEvalOutput] = None
        self.epoch = 0
        self.best_weights = None

        # Evaluate with initial loaded weights to check if everything works
        # set n_epochs_no_eval to 0 to skip this step
        if self.epoch > self.n_epochs_no_eval:
            self.dev_eval_output = self._eval_dev()
            if self.evaluate_train:
                self.train_eval_output = self._eval_train()
            if self.test:
                self.test_document_and_overlap_len_to_eval_outputs = self._test()
            max_dev_score = self.dev_eval_output[1].span_score
            self._log("\n")

        # Training loop
        self._log("Starting training\n")
        for self.epoch in range(1, self.max_epochs + 1):
            self._log(f"Epoch = {self.epoch}")
            self._train()
            if self.epoch > self.n_epochs_no_eval:
                self.dev_eval_output = self._eval_dev()
                self.dev_losses.append(self.dev_eval_output[0])
                self.dev_scores.append(self.dev_eval_output[1].span_score)
                if self.evaluate_train:
                    self.train_eval_output = self._eval_train()
                    self.train_losses.append(self.train_eval_output[0])
                    self.train_scores.append(self.train_eval_output[1].span_score)
                dev_score = self.dev_eval_output[1].span_score
                if dev_score > max_dev_score:
                    max_dev_score = dev_score
                    n_epochs_early_stopping = 0
                    self._save()
                    self.best_weights = copy.deepcopy(self.model.weights)
                else:
                    n_epochs_early_stopping += 1
                self._log(f"Dev score decreasing for {n_epochs_early_stopping} epochs, "
                          f"{self.patience - n_epochs_early_stopping} epochs left till early stop")
                if n_epochs_early_stopping == self.patience:
                    break
            self._log("\n")
        
        # Loss Curve
        self._save_plot()

        # Test
        if self.test:
            self._log("\nTesting")
            self.model.load_weights(self.best_weights)
            self.dev_eval_output = self._eval_dev()
            self.test_document_and_overlap_len_to_eval_outputs = self._test()
            if self.save_predictions:
                self._save_test_predictions()
            if self.save_log:
                self._save_config()

        # Clean
        self._clean()