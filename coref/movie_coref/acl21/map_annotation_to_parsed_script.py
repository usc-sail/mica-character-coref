import argparse
import numpy as np
from bisect import bisect_right
import pandas as pd
from tqdm import trange

def longest_common_subsequence_map(A, B):
    n, m = len(A), len(B)
    L = np.zeros((n + 1, m + 1), dtype=np.int)
    D = np.full((n + 1, m + 1), "-", dtype="<U1")
    
    print("finding LCS")
    for i in trange(1, n + 1):
        for j in range(1, m + 1):
            a = A[i - 1]
            b = B[j - 1]
            
            if a == b:
                L[i, j] = L[i - 1, j - 1] + 1
                D[i, j] = "d"
                
            elif L[i - 1, j] >= L[i, j - 1]:
                L[i, j] = L[i - 1, j]
                D[i, j] = "u"
                
            else:
                L[i, j] = L[i, j - 1]
                D[i, j] = "l"
                
    lcs_map = []
    d, i, j = D[n, m], n, m

    while d != "-":
        if d == "d":
            lcs_map.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d == "u":
            i -= 1
        else:
            j -= 1
        d = D[i, j]
    
    lcs_map = sorted(lcs_map)        
    return lcs_map

def map_coreference_annotation_to_parsed_script(script_file, parsed_file, coref_annotation_file, \
    mapped_coref_annotation_file):
    script = open(script_file).read()
    parsed = open(parsed_file).read().strip()
    coref_df = pd.read_csv(coref_annotation_file, index_col=None).sort_values(by="begin")
    coref_df.index = pd.RangeIndex(len(coref_df))

    i, tags, element_texts, element_begins, element_ends, parsed_script = 0, [], [], [], [], ""

    for line in parsed.split("\n"):
        tag, text = line[0], line[2:].strip()
        tags.append(tag)
        element_texts.append(text)
        element_begins.append(i)
        element_ends.append(i + len(text))
        parsed_script += text
        i += len(text)

    lcs_map = longest_common_subsequence_map(script, parsed_script)
    lcs_map_dict = dict(lcs_map)
    records = []

    for _, row in coref_df.iterrows():
        begin, end = row.begin, row.end - 1
        if begin in lcs_map_dict and end in lcs_map_dict:
            pbegin, pend = lcs_map_dict[begin], lcs_map_dict[end]
            i = bisect_right(element_begins, pbegin) - 1
            j = bisect_right(element_begins, pend) - 1
            pi = pbegin - element_begins[i]
            pj = pend - element_begins[j]
            records.append([pbegin, i, pi, pend, j, pj])
        else:
            records.append([np.nan] * 6)

    mapped_df = pd.DataFrame(records, columns = ["pbegin", "pbegin_ind", "pbegin_pos", "pend", "pend_ind", "pend_pos"])
    coref_mapped_df = pd.concat([coref_df, mapped_df], axis=1)

    parsed_mentions = []
    mentions = []

    for _, row in coref_mapped_df.iterrows():
        parsed_mention = None
        mention = script[row.begin: row.end]
        if pd.notna(row.pbegin) and row.pbegin_ind == row.pend_ind:
            i, j, k = int(row.pbegin_ind), int(row.pbegin_pos), int(row.pend_pos)
            parsed_mention = element_texts[i][j: k + 1]
        mentions.append(mention)
        parsed_mentions.append(parsed_mention)
        
    coref_mapped_df["parsed_mention"] = parsed_mentions
    coref_mapped_df["mention"] = mentions

    p = len(coref_mapped_df) 
    q = coref_mapped_df.pbegin.notna().sum()
    r = (coref_mapped_df.pbegin_ind == coref_mapped_df.pend_ind).sum()
    s = (coref_mapped_df.mention.str.replace("\s", "") == coref_mapped_df.parsed_mention.str.replace("\s", "")).sum()

    print(f"{p} coreference mentions, {q} mentions mapped ({100*q/p:.2f}%)")
    print(f"{r} mapped mentions belong to same parsed script element ({100*r/p:.2f}%)")
    print(f"{s} element and script mentions are equal ({100*s/p:.2f}%)")

    coref_mapped_df.to_csv(mapped_coref_annotation_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="map coreference annotations to parsed script")
    parser.add_argument("-a", help="annotation file", dest="ann", type=str)
    parser.add_argument("-s", help="script file", dest="script", type=str)
    parser.add_argument("-p", help="parsed file", dest="parsed", type=str)
    parser.add_argument("-o", help="mapped annotation file", dest="out", type=str)

    args = parser.parse_args()
    script_file = args.script
    parsed_file = args.parsed
    coref_annotation_file = args.ann
    mapped_coref_annotation_file = args.out

    map_coreference_annotation_to_parsed_script(script_file, parsed_file, coref_annotation_file, \
    mapped_coref_annotation_file)