"""Create sequence coreference pytorch tensors data for training and testing."""

from mica_text_coref.coref.seq_coref import data

from keras_preprocessing import sequence
import tqdm
import torch
from torch.utils import data as tdata
from transformers import LongformerTokenizer

def create_tensors(
    corpus: data.CorefCorpus, 
    representative_mentions: list[list[data.Mention]],
    longformer_tokenizer: LongformerTokenizer,
    max_sequence_length = 4096) -> (
        tdata.TensorDataset):
    """Create Tensor Dataset from coreference corpus and representative mentions
     for each document's cluster. The number of representative
    mentions should equal the number of clusters. Make sure that you remap
    spans of the document before passing it here.

    Args:
        corpus: Coreference corpus.
        representative_mentions: List of list of data.Mention objects.
        coref_longformer_model: Coreference Longformer Model.
    
    Returns:
        A tensor pytorch dataset. It contains the following tensors: 
            1. token ids: LongTensor
            2. mention ids: IntTensor
            3. label ids: IntTensor
            4. attention mask: FloatTensor
            5. global attention mask: FloatTensor
            6. doc ids: IntTensor
    """
    assert len(corpus.documents) == len(representative_mentions), (
        "Number of documents should equal the number of representative mention"
        " lists")
    for document, mentions in zip(corpus.documents, representative_mentions):
        assert len(document.clusters) == len(mentions), f"Number of clusters"
        " should equal the number of representative mentions,"
        " doc key = {document.doc_key}"

    _max_sequence_length = 0
    for document in corpus.documents:
        n_tokens = sum(len(sentence) for sentence in document.sentences)
        _max_sequence_length = max(n_tokens, _max_sequence_length)
    max_sequence_length = min(max_sequence_length, _max_sequence_length)

    token_ids_list: list[list[int]] = []
    mention_ids_list: list[list[int]] = []
    label_ids_list: list[list[int]] = []
    attn_mask_list: list[list[int]] = []
    global_attn_mask_list: list[list[int]] = []
    doc_ids: list[int] = []

    for i, document in tqdm.tqdm(enumerate(corpus.documents), desc="tensorize",
                                           total=len(corpus.documents)):
        tokens = [token for sentence in document.sentences 
                        for token in sentence]
        token_ids: list[int] = longformer_tokenizer.convert_tokens_to_ids(
            tokens)
        attn_mask: list[int] = [1 for _ in range(len(tokens))]
        doc_id = document.doc_id
        
        for j, cluster in enumerate(document.clusters):
            sorted_cluster = sorted(cluster)
            
            if sorted_cluster[-1].end < max_sequence_length:
                mention = representative_mentions[i][j]
                mention_ids = [0 for _ in range(len(tokens))]
                mention_ids[mention.begin] = 1
                for k in range(mention.begin + 1, mention.end + 1):
                    mention_ids[k] = 2
                label_ids = [0 for _ in range(len(tokens))]
                global_attn_mask = [0 for _ in range(len(tokens))]
                for k in range(mention.begin, mention.end + 1):
                    global_attn_mask[k] = 1
                
                for mention in sorted_cluster:
                    label_ids[mention.begin] = 1
                    for k in range(mention.begin + 1, mention.end + 1):
                        label_ids[k] = 2
                
                token_ids_list.append(token_ids)
                mention_ids_list.append(mention_ids)
                label_ids_list.append(label_ids)
                attn_mask_list.append(attn_mask)
                global_attn_mask_list.append(global_attn_mask)
                doc_ids.append(doc_id)
    
    token_ids_pt = torch.LongTensor(sequence.pad_sequences(token_ids_list, 
        maxlen=max_sequence_length, dtype=int, padding="post", 
        truncating="post", value=longformer_tokenizer.pad_token_id))
    mention_ids_pt = torch.IntTensor(sequence.pad_sequences(mention_ids_list,
        maxlen=max_sequence_length, dtype=int, padding="post",
        truncating="post", value=0))
    label_ids_pt = torch.IntTensor(sequence.pad_sequences(label_ids_list,
        maxlen=max_sequence_length, dtype=int, padding="post", 
        truncating="post", value=0))
    attn_mask_pt = torch.FloatTensor(sequence.pad_sequences(attn_mask_list, 
        maxlen=max_sequence_length, dtype=float, padding="post",
        truncating="post", value=0.))
    global_attn_mask_pt = torch.FloatTensor(sequence.pad_sequences(
        global_attn_mask_list, 
        maxlen=max_sequence_length, dtype=float, padding="post",
        truncating="post", value=0.))
    doc_ids_pt = torch.IntTensor(doc_ids)

    dataset = tdata.TensorDataset(token_ids_pt, mention_ids_pt, label_ids_pt, 
                                attn_mask_pt, global_attn_mask_pt, doc_ids_pt)
    return dataset