# pyright: reportGeneralTypeIssues=false
from mica_text_coref.coref.movie_coref import split_and_merge
import os
import jsonlines
import numpy as np

preprocess_arr = ["regular", "addsays", "nocharacters"]
split_len_arr = [2048, 3072, 4096, 5120]
overlap_len_arr = [128, 256, 512]

for preprocess in preprocess_arr:
    for split_len in split_len_arr:
        for overlap_len in overlap_len_arr:
            path = os.path.join(os.getenv("DATA_DIR"), f"mica_text_coref/movie_coref/results/{preprocess}/train_wl.jsonlines")
            subdoc_lens, subdoc_overlap_lens, nz = [], [], 0
            with jsonlines.open(path) as reader:
                for doc in reader:
                    prev = None
                    for subdoc in split_and_merge.split_screenplay(doc, split_len, overlap_len):
                        subdoc_len = len(subdoc["cased_words"])
                        subdoc_lens.append(subdoc_len)
                        if prev is not None:
                            overlap = prev - subdoc["offset"][0]
                            subdoc_overlap_lens.append(overlap)
                            if overlap == 0:
                                nz += 1
                        prev = subdoc["offset"][1]
            print(f"preprocess={preprocess:12s} split_len={split_len:4d} overlap_len={overlap_len:3d}: avg subdoc len = {np.mean(subdoc_lens):5.1f} (±{np.std(subdoc_lens):4.1f}) "
                f"avg overlap len = {np.mean(subdoc_overlap_lens):4.1f} (±{np.std(subdoc_overlap_lens):3.1f}) #zeros = {nz:2d}")
