"""Train wl-roberta model on movie screenplay coreference data in a distributed
system using huggingface accelerate library.
"""
from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference
from mica_text_coref.coref.movie_coref.data import CorefCorpus, CorefDocument, Mention, GraphNode
from mica_text_coref.coref.movie_coref.data import parse_labelset, pos_labelset, ner_labelset
from mica_text_coref.coref.movie_coref.result import CorefResult

import accelerate
from accelerate import logging
import bisect
import collections
import gpustat
import itertools
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
        output_dir: str,
        reference_scorer: str,
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
        character_lr: float,
        coref_lr: float,
        weight_decay: float,
        max_epochs: int,
        train_document_len: int,
        eval_document_len: int,
        eval_document_overlap_len: int,
        cr_seq_len: int,
        subword_batch_size: int,
        cr_batch_size: int,
        fn_batch_size: int,
        run_span: bool,
        add_cr_to_coarse: bool,
        filter_mentions_by_cr: bool,
        remove_singleton_cr: bool,
        train_cr_epochs: int,
        train_bert_with_cr: bool,
        save_model: bool,
        save_output: bool,
        save_loss_curve: bool,
        debug: bool
        ) -> None:
        self.accelerator = accelerate.Accelerator()
        self.logger = logging.get_logger("")
        self._add_log_file(log_file)
        self.output_dir = output_dir
        self.reference_scorer = reference_scorer
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
        self.character_lr = character_lr
        self.coref_lr = coref_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.train_document_len = train_document_len
        self.eval_document_len = eval_document_len
        self.eval_document_overlap_len = eval_document_overlap_len
        self.cr_seq_len = cr_seq_len
        self.subword_batch_size = subword_batch_size
        self.cr_batch_size = cr_batch_size
        self.fn_batch_size = fn_batch_size
        self.run_span = run_span
        self.add_cr_to_coarse = add_cr_to_coarse
        self.filter_mentions_by_cr = filter_mentions_by_cr
        self.remove_singleton_cr = remove_singleton_cr
        self.train_cr_epochs = train_cr_epochs
        self.train_bert_with_cr = train_bert_with_cr
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

    def _create_optimizers(self):
        """Create and accelerate optimizers. If freeze_bert is true, don't create the bert
        optimizer
        """
        # Create optimizers
        self._log("Creating optimizers")
        if self.freeze_bert:
            for param in self.model.bert_parameters():
                param.requires_grad = False
        else:
            self.bert_optimizer = AdamW(self.model.bert_parameters(), lr=self.bert_lr, 
                weight_decay=self.weight_decay)
        self.cr_optimizer = AdamW(self.model.cr_parameters(), lr=self.character_lr, 
            weight_decay=self.weight_decay)
        self.coref_optimizer = AdamW(self.model.coref_parameters(), lr=self.coref_lr, 
            weight_decay=self.weight_decay)

        # Accelerate model
        self._log("Accelerating model and optimizers")
        for module in self.model.modules():
            if next(module.parameters()).requires_grad:
                module = self.accelerator.prepare_model(module)
            else:
                module.to(self.accelerator.device)

        # Accelerate optimizer
        if self.freeze_bert:
            self.cr_optimizer, self.coref_optimizer = self.accelerator.prepare(self.cr_optimizer, 
                self.coref_optimizer)
        else:
            self.bert_optimizer, self.cr_optimizer, self.coref_optimizer = (
                self.accelerator.prepare(self.bert_optimizer, self.cr_optimizer, 
                    self.coref_optimizer))

    def _split_screenplay(
        self, document: CorefDocument, split_len: int, overlap_len: int):
        """Split screenplay document into smaller documents.

        Args:
            document: CorefDocument object representing the original screenplay
                document.
            split_len: Length of the smaller CorefDocument objects in words
                the original screenplat is split into.
            overlap_len: Number of words overlapping between successive smaller
                CorefDocuments.
        
        Returns:
            Generator of CorefDocument objects.
        """
        doc_offsets: list[tuple[int, int]] = []
        segment_boundaries = np.zeros(len(document.token), dtype=int)
        i = 0
        parse_tags = document.parse
        sentence_offsets = np.array(document.sentence_offsets)
        while i < len(document.token):
            if parse_tags[i] in "SNC":
                j = i + 1
                while j < len(document.token) and (
                    parse_tags[j] == parse_tags[i]):
                    j += 1
                segment_boundaries[i] = 1
                i = j
            else:
                i += 1
        segment_boundaries[0] = 1
        i = 0
        while i < len(document.token):
            j = min(i + split_len, len(document.token))
            if j < len(document.token):
                while j >= i and segment_boundaries[j] == 0:
                    j -= 1
                k = i + split_len - overlap_len
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
            _document.sentence_offsets = (sentence_offsets[si: sj] - sentence_offsets[si, 0]
                ).tolist()
            clusters: dict[str, set[Mention]] = collections.defaultdict(set)
            n_mentions = 0
            for character, mentions in document.clusters.items():
                for mention in mentions:
                    assert (mention.end < i or i <= mention.begin <= mention.end < j or 
                        j <= mention.begin), "Mention crosses document boundaries"
                    if i <= mention.begin <= mention.end < j:
                        mention.begin -= i
                        mention.end -= i
                        mention.head -= i
                        clusters[character].add(mention)
                        n_mentions += 1
            _document.clusters = clusters
            _document.word_cluster_ids = document.word_cluster_ids[i: j]
            _document.word_head_ids = document.word_head_ids[i: j]
            self._log(f"{_document.movie}: {len(_document.token)} words, "
                f"{n_mentions} mentions, {len(_document.clusters)} clusters")
            yield _document

    def _tokenize_document(self, document: CorefDocument):
        """Tokenize the words of the document into subwords.
        """
        words = document.token
        subword_ids = []
        word_to_subword_offset = []
        for word in words:
            _subwords = self.model.tokenizer_map.get(word, self.model.tokenizer.tokenize(word))
            word_to_subword_offset.append([len(subword_ids), len(subword_ids) + len(_subwords)])
            subword_ids.extend(self.model.tokenizer.convert_tokens_to_ids(_subwords))
        document.subword_ids = subword_ids
        document.word_to_subword_offset = word_to_subword_offset
        self._log(f"{document.movie}: {len(subword_ids)} subwords, "
            f"{len(word_to_subword_offset)} word to subword offsets")

    def _create_subword_dataloader(self, document: CorefDocument):
        """Create dataloader of subword sequences and mask.
        """
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
            subword_id_seqs.append([cls_id] + subword_ids[i : j] + [sep_id] + [pad_id] * (
                L - j + i))
            subword_mask_seqs.append([0] + [1] * (j - i) + [0] * (L - j + i + 1))
            i = j
        self._log(f"{document.movie}: {len(subword_id_seqs)} subword sequences, {L} "
            f"subwords/sequence")
        subword_id_tensor = torch.LongTensor(subword_id_seqs)
        subword_mask_tensor = torch.FloatTensor(subword_mask_seqs)
        document.subword_dataloader = self.accelerator.prepare_data_loader(DataLoader(TensorDataset(
            subword_id_tensor, subword_mask_tensor), batch_size=self.subword_batch_size))
        batch_size = math.ceil(len(subword_id_seqs)/len(document.subword_dataloader))
        self._log(f"{document.movie}: Subword batch size = {batch_size} sequences/batch")

    def _prepare_corpus(
        self, corpus: CorefCorpus, split_len: int | None, 
        overlap_len = 0) -> CorefCorpus:
        """Create new corpus by splitting, tokenizing, and creating subword
        sequences of the screenplay documents in the original corpus.

        Args:
            corpus: CorefCorpus containing screenplay documents.
            split_len: Length of the new documents in words the screenplay
                documents will be split into. If split_len is None, then no
                splitting occurs.
            overlap_len: Number of words overlapping between successive new
                documents. Ignored if no splitting occurs.
        """
        _corpus = CorefCorpus()
        if split_len is None:
            document_generator = corpus
        else:
            document_generator = itertools.chain(*[self._split_screenplay(document, split_len, 
                overlap_len) for document in corpus])
        for _document in document_generator:
            self._tokenize_document(_document)
            self._create_subword_dataloader(_document)
            _corpus.documents.append(_document)
        return _corpus
    
    def _find_avg_n_heads(self, corpus: CorefCorpus) -> float:
        """Get average number of character mention heads in a document.
        """
        n_heads = []
        for document in corpus:
            n_document_heads = 0
            for mentions in document.clusters.values():
                n_document_heads += len(mentions)
            n_heads.append(n_document_heads)
        return np.mean(n_heads)

    def _gather(self, dataloader: DataLoader, batch_size: int):
        """Get the appropriate gather function (gather or gather_for_metrics)
        given the dataloader and batch size.
        TODO: raise issue at accelerator github.
        """
        n_samples = len(dataloader.dataset)
        n_processes = self.accelerator.num_processes
        if n_samples % (batch_size * n_processes) == 0:
            return self.accelerator.gather
        else:
            return self.accelerator.gather_for_metrics
    
    def _subword_gather(self, dataloader: DataLoader):
        """Get gather function to use in subword dataloader"""
        return self._gather(dataloader, self.subword_batch_size)
    
    def _cr_gather(self, dataloader: DataLoader):
        """Get gather function to use in character head dataloader"""
        return self._gather(dataloader, self.cr_batch_size)

    def _fn_gather(self, dataloader: DataLoader):
        """Get gather function to use in fine coreference dataloader"""
        return self._gather(dataloader, self.fn_batch_size)

    def _create_cr_dataloader(
        self, word_embeddings: torch.FloatTensor, 
        document: CorefDocument) -> DataLoader:
        """Get dataloader for character head recognition model.

        Args:
            word_embeddings: Float tensor of word embeddings 
                [n_words, embedding_size]
            document: CorefDocument
        
        Returns:
            Tensor Dataloader
        """
        seq_parsetag_ids = []
        seq_postag_ids = []
        seq_nertag_ids = []
        seq_ispronoun = []
        seq_ispunctuation = []
        n_seqs = math.ceil(len(word_embeddings)/self.cr_seq_len)
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        for i in range(n_seqs):
            _seq_parsetag_ids = document.parse_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_postag_ids = document.pos_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_nertag_ids = document.ner_ids[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispronoun = document.is_pronoun[i * self.cr_seq_len: (i + 1) * self.cr_seq_len]
            _seq_ispunctuation = document.is_punctuation[i * self.cr_seq_len: 
                (i + 1) * self.cr_seq_len]
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
        tensor_word_embeddings = torch.cat([word_embeddings, torch.zeros(
                (n_seqs * self.cr_seq_len - len(word_embeddings), word_embeddings.shape[1]), 
                device=self.accelerator.device)], dim=0).view((n_seqs, self.cr_seq_len, -1))
        tensor_parsetag_ids = torch.LongTensor(seq_parsetag_ids).to(self.accelerator.device)
        tensor_postag_ids = torch.LongTensor(seq_postag_ids).to(self.accelerator.device)
        tensor_nertag_ids = torch.LongTensor(seq_nertag_ids).to(self.accelerator.device)
        tensor_ispronoun = torch.LongTensor(seq_ispronoun).to(self.accelerator.device)
        tensor_ispunctuation = torch.LongTensor(seq_parsetag_ids).to(self.accelerator.device)
        self._debug(f"{prefix}: {len(tensor_word_embeddings)} cr word embedding "
            f"sequences, {tensor_word_embeddings.shape[1]} words/sequence")
        dataloader = self.accelerator.prepare_data_loader(DataLoader(TensorDataset(
            tensor_word_embeddings, tensor_parsetag_ids, tensor_postag_ids, tensor_nertag_ids, 
            tensor_ispronoun, tensor_ispunctuation), batch_size=self.cr_batch_size))
        batch_size = math.ceil(len(tensor_word_embeddings)/len(dataloader))
        self._debug(f"{prefix}: CR batch size = {batch_size} sequences/batch")
        return dataloader

    def _create_fine_dataloader(
        self, word_embeddings: torch.FloatTensor, features: torch.FloatTensor,
        scores: torch.FloatTensor, indices: torch.LongTensor,
        document: CorefDocument) -> DataLoader:
        """Get dataloader for fine coreference scoring.

        Args:
            word_embeddings: Float tensor of word embeddings 
                [n_words, embedding_size]
            features: Float tensor of pairwise features (genre, same speaker, 
                and distance) of words and its top k scoring antecedents.
                [n_words, n_antecedents, feature_size]
            scores: Float tensor of coarse coreference scores of words and its
                top k scoring antecedents. [n_words, n_antecedents]
            indices: Long tensor of the word indices of the top scoring 
                antecedents. [n_words, n_antecedents]. 
                Lies between 0 and n_words - 1
            document: CorefDocument object.
        
        Returns:
            Torch dataloader.
        """
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie}"
        dataset = TensorDataset(word_embeddings, features, indices, scores)
        dataloader = DataLoader(dataset, batch_size=self.fn_batch_size)
        dataloader = self.accelerator.prepare_data_loader(dataloader)
        batch_size = math.ceil(len(word_embeddings)/len(dataloader))
        self._debug(f"{prefix}: fn batch size = {batch_size} word pairs/batch")
        return dataloader

    def _get_coref_ground_truth(
        self, cluster_ids: list[int], top_indices: torch.LongTensor, 
        valid_pair_map: torch.Tensor) -> torch.FloatTensor:
        """Get coreference ground truth for evaluation.

        Args:
            cluster_ids: List of word cluster ids. 0 if word is not a character
                head, >=1 if word is a character head. Co-referring words have
                same cluster id. [n_words]
            top_indices: Long tensor of top scoring antecedents.
                [n_words, n_antecedents]
            valid_pair_mask: Bool tensor of whether the word-antecedent pair is
                valid (word comes after antecedents). [n_words, n_antecedents]
        
        Returns:
            Coreference labels y. 
            Float tensor of shape [n_words, n_antecedents + 1].
            y[i, j + 1] = 1, if the ith word co-refers with its jth 
                antecedent, else 0, for j = 0 to n_antecedents - 1
            y[i, 0] = 1, if for all j = 0 to n_antecedents - 1 
                y[i, j + 1] = 0, else 0
        """
        cluster_ids = torch.Tensor(cluster_ids).to(self.accelerator.device)
        y = cluster_ids[top_indices] * valid_pair_map
        y[y == 0] = -1
        dummy = torch.zeros((len(y), 1), dtype=y.dtype, device=self.accelerator.device)
        y = torch.cat([dummy, y], dim=1)
        y = (y == cluster_ids.unsqueeze(1))
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    def _get_word_clusters(
        self, document: CorefDocument, character_scores: torch.FloatTensor, 
        coref_scores: torch.FloatTensor, indices: torch.LongTensor
        ) -> tuple[list[set[int]], list[set[int]]]:
        """
        Find word-level clusters from character head and coreference scores.
        Return both gold and predicted clusters.

        Args:
            character_scores: Float tensor of word-level logits of the word
                being the head of a character mention. [n_words]
            coref_scores: Float tensor of logits of the word pair being
                coreferent with each other. The first column contains all zeros. 
                [n_words, 1 + n_antecedents]
            indices: Long tensor of antecedent indices. [n_words, n_antecedents]
        
        Returns:
            list[set[int]], list[set[int]]
        """
        is_character = (torch.sigmoid(character_scores) > 0.5).tolist()
        antecedents = coref_scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(coref_scores))[not_dummy]
        antecedents = indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(coref_scores))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            if not self.filter_mentions_by_cr or (is_character[i] and is_character[j]):
                nodes[i].link(nodes[j])
                assert nodes[i] is not nodes[j]

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

        gold_clusters = []
        for mentions in document.clusters.values():
            cluster = set([])
            for mention in mentions:
                cluster.add(mention.head)
            gold_clusters.append(cluster)

        return gold_clusters, pred_clusters

    def _get_sp_training_data(self, document: CorefDocument) -> (
        tuple[
            torch.LongTensor, torch.LongTensor, torch.LongTensor, 
            torch.LongTensor]):
        """Get head, start, and end word indexes from document for training the
        span prediction module.

        Returns:
            parse_ids: Long tensor of word screenplay parse tags [n_words]
            heads: Long tensor of word indices of heads of character mentions 
                [n_heads]
            starts: Long tensor of word indices of beginning of character
                mentions [n_heads]
            ends: Long tensor of word indices of ending of character
                mentions [n_heads]
        """
        parse_ids = torch.LongTensor(document.parse_ids).to(self.accelerator.device)
        starts, ends, heads = [], [], []
        for mentions in document.clusters.values():
            for mention in mentions:
                starts.append(mention.begin)
                ends.append(mention.end)
                heads.append(mention.head)
        starts = torch.LongTensor(starts).to(self.accelerator.device)
        ends = torch.LongTensor(ends).to(self.accelerator.device)
        heads = torch.LongTensor(heads).to(self.accelerator.device)
        return parse_ids, heads, starts, ends
    
    def _get_sp_inference_data(
        self, document: CorefDocument, word_clusters: list[list[int]]) -> (
            tuple[torch.LongTensor, torch.LongTensor]):
        """Get parse ids and heads from the predicted word clusters.

        Returns:
            parse_ids: Long tensor of word screenplay parse tags [n_words]
            heads: Long tensor of word indices of heads of predicted
                character mentions [n_heads]
        """
        parse_ids = torch.LongTensor(document.parse_ids).to(self.accelerator.device)
        heads = set([_head for cluster in word_clusters for _head in cluster])
        heads = sorted(heads)
        heads = torch.LongTensor(heads).to(self.accelerator.device)
        return parse_ids, heads

    def _run(self, document: CorefDocument) -> tuple[torch.Tensor, CorefResult]:
        """Model coreference in document.

        Returns:
            loss: Training loss to be used in backprop.
            coref_result: CorefResult object containing gold and predicted
                word clusters, character heads, and span clusters.
        
        In evaluation, the loss does not contain span prediction loss.
        """
        prefix = f"Epoch = {self.epoch:2d}, Movie = {document.movie:20s}"
        result = CorefResult(self.reference_scorer, self.output_dir, self.epoch)

        # Bert
        subword_embeddings = []
        subword_gather = self._subword_gather(document.subword_dataloader)
        for batch in document.subword_dataloader:
            ids, mask = batch
            embeddings = self.model.bert(ids, mask).last_hidden_state
            embeddings, mask = subword_gather((embeddings, mask))
            embeddings = embeddings[mask == 1]
            subword_embeddings.append(embeddings)
            self.accelerator.wait_for_everyone()
        subword_embeddings = torch.cat(subword_embeddings, dim=0)
        self._debug(f"{prefix}: subword_embeddings = {subword_embeddings.shape} "
            f"({subword_embeddings.device})")
        self._debug(self._gpu_usage())

        # Word Encoder
        word_to_subword_offset = torch.LongTensor(document.word_to_subword_offset).to(
            self.accelerator.device)
        word_embeddings = self.model.encoder(subword_embeddings, word_to_subword_offset)
        self._debug(f"{prefix}: word_embeddings = {word_embeddings.shape} "
            f"({word_embeddings.device})")
        self._debug(self._gpu_usage())

        # Character Scores
        character_scores = []
        cr_dataloader = self._create_cr_dataloader(word_embeddings, document)
        cr_gather = self._cr_gather(cr_dataloader)
        for batch in cr_dataloader:
            embeddings = batch[0]
            scores = self.model.character_recognizer(*batch)
            embeddings, scores = cr_gather((embeddings, scores))
            scores = scores[~(embeddings == 0).all(dim=2)]
            character_scores.append(scores)
            self.accelerator.wait_for_everyone()
        character_scores = torch.cat(character_scores, dim=0)
        self._debug(f"{prefix}: character_scores = {character_scores.shape} "
            f"({character_scores.device})")
        self._debug(self._gpu_usage())

        # Coarse Coreference Scores
        if self.add_cr_to_coarse:
            coarse_scores, top_indices = self.model.coarse_scorer(word_embeddings, character_scores)
        else:
            coarse_scores, top_indices = self.model.coarse_scorer(word_embeddings)
        self._debug(f"{prefix}: coarse_scores = {coarse_scores.shape}, top_indices = "
            f"{top_indices.shape} ({top_indices.device})")
        torch.cuda.empty_cache()
        self._debug(self._gpu_usage())

        # Pairwise Encoder
        features = self.model.pairwise_encoder(top_indices, document.speaker, self.genre)
        self._debug(f"{prefix}: features = {features.shape} ({features.device})")
        self._debug(self._gpu_usage())
        
        # Fine Coreference Scores
        fine_scores = []
        fn_dataloader = self._create_fine_dataloader(word_embeddings, features, coarse_scores, 
            top_indices, document)
        fn_gather = self._fn_gather(fn_dataloader)
        for batch in fn_dataloader:
            scores = self.model.fine_scorer(word_embeddings, *batch)
            scores = fn_gather(scores)
            fine_scores.append(scores)
            self.accelerator.wait_for_everyone()
        fine_scores = torch.cat(fine_scores, dim=0)
        self._debug(f"{prefix}: fine_scores = {fine_scores.shape} ({fine_scores.device})")
        torch.cuda.empty_cache()
        self._debug(self._gpu_usage())

        # Loss
        coref_y = self._get_coref_ground_truth(document.word_cluster_ids, top_indices,
            (coarse_scores > float("-inf")))
        coref_loss = self.model.coref_loss(fine_scores, coref_y)
        character_y = torch.FloatTensor(document.word_head_ids).to(self.accelerator.device)
        character_loss = self.model.cr_loss(character_scores, character_y)
        loss = coref_loss + character_loss
        torch.cuda.empty_cache()
        self._log(f"{prefix}: character_loss = {character_loss:.4f}, coref_loss = "
            f"{coref_loss:.4f}, loss (character + coref) = {loss:.4f}")
        self._debug(self._gpu_usage())

        # Performance
        gold_word_clusters, pred_word_clusters = self._get_word_clusters(document, character_scores,
            fine_scores, top_indices)
        gold_character_heads = document.word_head_ids
        pred_character_heads = (torch.sigmoid(character_scores) > 0.5).to(torch.long).tolist()
        result.add_word_clusters(document, gold_word_clusters, pred_word_clusters)
        result.add_characters(document, gold_character_heads, pred_character_heads)

        # Span Prediction
        if self.run_span:
            if self.model.training:
                parse_ids, heads, starts, ends = self._get_sp_training_data(document)
                sp_scores = self.model.span_predictor(word_embeddings, parse_ids, heads)
                sp_loss = self.model.sp_loss(sp_scores, starts, ends, self.avg_n_train_heads)
                loss = loss + sp_loss
                self._debug(f"{prefix}: heads = {heads.shape}, sp_scores = {sp_scores.shape} "
                    f"({sp_scores.device})")
                self._log(f"{prefix}: span_loss = {sp_loss:.4f}, loss (character + coref + span)"
                    f" = {loss:.4f}")
                torch.cuda.empty_cache()
                self._debug(self._gpu_usage())
            else:
                parse_ids, heads = self._get_sp_inference_data(document, pred_word_clusters)
                sp_scores = self.model.span_predictor(word_embeddings, parse_ids, heads)
                starts = sp_scores[:, :, 0].argmax(dim=1).tolist()
                ends = sp_scores[:, :, 1].argmax(dim=1).tolist()
                head2span = {head: (start, end) for head, start, end in zip(heads.tolist(), starts,
                    ends)}
                pred_span_clusters = [set([head2span[head] for head in cluster])
                    for cluster in pred_word_clusters]
                gold_span_clusters = [set([(mention.begin, mention.end) for mention in cluster])
                    for cluster in document.clusters]
                result.add_span_clusters(document, gold_span_clusters, pred_span_clusters)

        return loss, result

    def _step(self):
        """Update model weights"""
        self.cr_optimizer.step()
        if self.freeze_bert:
            if self.epoch > self.train_cr_epochs:
                self.coref_optimizer.step()
        else:
            if self.epoch <= self.train_cr_epochs:
                if self.train_bert_with_cr:
                    self.bert_optimizer.step()
            else:
                self.bert_optimizer.step()
                self.coref_optimizer.step()

    def _train(self, document: CorefDocument) -> float:
        """Train model on document and return loss value.
        """
        self.model.train()
        if not self.freeze_bert:
            self.bert_optimizer.zero_grad()
        self.cr_optimizer.zero_grad()
        self.coref_optimizer.zero_grad()
        loss, _ = self._run(document)
        self.accelerator.backward(loss)
        self._step()
        self.accelerator.wait_for_everyone()
        return loss.item()
    
    def _eval(self):
        """Evaluate model on development corpus documents.
        """
        self.model.eval()
        with torch.no_grad():
            result = CorefResult(self.reference_scorer, self.output_dir, self.epoch)
            loss = []
            for document in self.dev_corpus:
                _loss, _result = self._run(document)
                loss.append(_loss.item())
                result.add(_result)
            self._log(f"Epoch = {self.epoch:2d}, eval average loss: {np.mean(loss):.4f}, "
                f"result = {result}")

    def __call__(self):
        self._debug(self._gpu_usage())

        # Create model
        self._log("Initializing model")
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

        # Load model weights
        self._log("Loading model weights from word-level-coref model")
        self.model.load_weights(self.weights_path)

        # Create optimizers
        self._create_optimizers()

        # Load training set
        self._log("Loading training corpus")
        self.train_corpus = CorefCorpus(self.train_path)
        self.train_corpus = self._prepare_corpus(self.train_corpus, self.train_document_len, 0)
        self.avg_n_train_heads = self._find_avg_n_heads(self.train_corpus)

        # Load development set
        self._log("Loading development corpus")
        self.dev_corpus = CorefCorpus(self.dev_path)
        self.dev_corpus = self._prepare_corpus(self.dev_corpus, self.eval_document_len, 
            self.eval_document_overlap_len)

        # Sanity check for eval
        self.epoch = 0
        self._eval()

        # Training loop
        self._log("Starting training")
        self._debug(self._gpu_usage())
        for self.epoch in range(1, self.max_epochs + 1):
            # TODO randomize this
            inds = np.arange(len(self.train_corpus))
            training_losses = []
            for self.doc_index in inds:

                # Train 
                document = self.train_corpus[self.doc_index]
                self._debug_all(f"Epoch = {self.epoch:2d}, Movie = {document.movie} "
                    f"({self.accelerator.device})")
                train_loss = self._train(document)
                training_losses.append(train_loss)
                torch.cuda.empty_cache()
                self._debug(self._gpu_usage())
                self._log(f"Epoch = {self.epoch:2d}: Running train loss = "
                    f"{np.mean(training_losses):.4f}")

            # Eval
            self._eval()