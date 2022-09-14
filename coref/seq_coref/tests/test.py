"""Test functions."""

from mica_text_coref.coref.seq_coref import coref_longformer

import os
import torch
from torch.utils import data as tdata

def load_tensors(directory) -> tdata.TensorDataset:
    token_ids = torch.load(os.path.join(directory, "tokens.pt"))
    mention_ids = torch.load(os.path.join(directory, "mentions.pt"))
    label_ids = torch.load(os.path.join(directory, "labels.pt"))
    attn_mask = torch.load(os.path.join(directory, "attn.pt"))
    global_attn_mask = torch.load(os.path.join(directory, "global_attn.pt"))
    doc_ids = torch.load(os.path.join(directory, "docs.pt"))
    dataset = tdata.TensorDataset(token_ids, mention_ids, label_ids,
                                    attn_mask, global_attn_mask, doc_ids)
    return dataset

dataset = load_tensors("/home/sbaruah_usc_edu/mica_text_coref/data/tensors/"
                       "longformer_seq_tensors/dev")
dataloader = tdata.DataLoader(dataset, batch_size=256, shuffle=True, 
                              drop_last=False)
model = coref_longformer.CorefLongformerModel()
for batch in dataloader:
    (batch_token_ids, batch_mention_ids, batch_label_ids, batch_attn_mask, 
        batch_global_attn_mask, batch_doc_ids) = batch
    break

document = """Entertainment media, available in rich variety and diverse forms ranging from traditional feature films, television shows, and theatrical plays to contemporary digital shorts and streaming content, can profoundly impact audience perceptions, beliefs, attitudes, and behavior.
Media narratives aim to inform and engage us with stories about the culture, lives, and experiences of different communities of people, including reflecting societal ideas and trends.
They shed light on various social, economic and political issues, educating and creating awareness on different aspects of life."""
long_document = " ".join(document for _ in range(50))
tokens = model.tokenizer.tokenize(long_document)[:4096]
token_ids = model.tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.LongTensor(token_ids)
attention_mask = torch.FloatTensor([1. for _ in range(len(tokens))])
global_attention_mask = torch.FloatTensor([0. for _ in range(len(tokens))])
batch_token_ids = token_ids.unsqueeze(0)
batch_attention_mask = attention_mask.unsqueeze(0)
batch_global_attention_mask = global_attention_mask.unsqueeze(0)
output = model.longformer(batch_token_ids, batch_attention_mask, 
                            batch_global_attention_mask)
print(output.shape, output.dtype)

longformer_output = model.longformer(batch_token_ids, batch_attn_mask, 
                                     batch_global_attn_mask)