import re
import spacy
import numpy as np
from textdistance import lcsseq
from collections import defaultdict

def find_speaker_clusters(speakers, min_sim, max_merges):
    normalized_speakers = [re.sub("\s+", "", speaker).lower() for speaker in speakers]
    unique_speakers = list(set(normalized_speakers))
    clusters = [[0, speaker, set()] for speaker in unique_speakers]
    
    for i, speaker in enumerate(normalized_speakers):
        j = unique_speakers.index(speaker)
        clusters[j][2].add(i)
        
    clusters = sorted(clusters, key = lambda cluster: len(cluster[2]), reverse=True)
    
    i = 0
    while i < len(clusters):
        js = []
        ri = clusters[i][1]
        
        for j in range(i + 1, len(clusters)):
            rj = clusters[j][1]
            n = len(lcsseq(ri, rj))
            d = len(ri) + len(rj)
            sim = 2*n/d
            if sim >= min_sim:
                js.append(j)
                
        cluster = clusters[i]
        merged_js = []
        
        for j in reversed(js):
            cluster[2].update(clusters[j][2])
            cluster[0] += 1
            merged_js.append(j)
            if cluster[0] >= max_merges:
                break
                
        clusters = clusters[:i] + [cluster] + [clusters[j] for j in range(i + 1, len(clusters)) if j not in merged_js]
        i += 1
        
    clusters = [cluster[2] for cluster in clusters]
    clusters = sorted(clusters, key = lambda cluster: len(cluster), reverse = True)
    speaker_clusters = [[speakers[i] for i in cluster] for cluster in clusters if len(cluster) >= 2]
    return clusters, speaker_clusters

def heuristic_pronoun_resolution(coref_tags, structure_tags, spacy_document):
    first_person_pronouns = "I, me, my, mine, myself".lower().split(", ")
    second_person_pronouns = "you, your, yours, yourself, yourselves".lower().split(", ")
    
    first_person_pronoun_tags = [token.text.lower() in first_person_pronouns for token in spacy_document]
    second_person_pronoun_tags = [token.text.lower() in second_person_pronouns for token in spacy_document]

    for i in range(len(coref_tags)):
        if first_person_pronoun_tags[i] and structure_tags[i] == "D":
            j = i - 1
            while j >= 0 and (structure_tags[j] == "D" or structure_tags[j] == "E" or structure_tags[j] == "X"):
                j -= 1
            if structure_tags[j] == "C":
                coref_tags[i] = coref_tags[j]
        
        elif second_person_pronoun_tags[i] and structure_tags[i] == "D":
            j, k = i - 1, i + 1
            prev_speaker = -1
            curr_speaker = -1
            next_speaker = -1

            while j >= 0 and structure_tags[j] in ["D","E","X"]:
                j -= 1
            if structure_tags[j] == "C":
                curr_speaker = coref_tags[j]
                j -= 1
                while j >= 0 and structure_tags[j] == "C":
                    j -= 1
                while j >= 0 and structure_tags[j]  in ["D","E","X"]:
                    j -= 1
                if structure_tags[j] == "C":
                    prev_speaker = coref_tags[j]
            
            while k < len(structure_tags) and structure_tags[k]  in ["D","E","X"]:
                k += 1
            if k < len(structure_tags) and structure_tags[k] == "C":
                next_speaker = coref_tags[k]

            if curr_speaker != -1:
                if prev_speaker != -1 and prev_speaker != curr_speaker:
                    coref_tags[i] = prev_speaker
                elif next_speaker != -1 and next_speaker != curr_speaker:
                    coref_tags[i] = next_speaker

    return coref_tags

def heuristic_coreference_resolution(document, structure_tags, spacy_document=None, min_speaker_sim=0.6, max_speaker_merges=3, pronoun_resolution=False):
    if spacy_document is None:
        spacy_nlp = spacy.load("en_core_web_sm")
        spacy_document = spacy_nlp(document)

    coref_tags = np.full(len(structure_tags), -1, dtype=np.int)
    speakers = []
    speaker_spans = []

    i = 0
    while i < len(spacy_document):
        if structure_tags[i] == "C":
            j = i
            while structure_tags[j] == "C":
                j += 1
            begin = spacy_document[i].idx
            end = spacy_document[j - 1].idx + len(spacy_document[j - 1])
            speaker = document[begin: end]
            speakers.append(speaker)
            speaker_spans.append((i, j - 1))
            i = j
        else:
            i += 1

    speaker_id_clusters, _ = find_speaker_clusters(speakers, min_speaker_sim, max_speaker_merges)
    
    for i, cluster in enumerate(speaker_id_clusters):
        for j in cluster:
            k, l = speaker_spans[j]
            for m in range(k, l + 1):
                coref_tags[m] = i

    if pronoun_resolution:
        coref_tags = heuristic_pronoun_resolution(coref_tags, structure_tags, spacy_document)

    clusters_dict = defaultdict(set)
    i = 0

    while i < len(coref_tags):
        if coref_tags[i] != -1:
            j = i + 1
            while j < len(coref_tags) and coref_tags[j] == coref_tags[i]:
                j += 1
            clusters_dict[coref_tags[i]].add((i, j - 1))
            i = j
        else:
            i += 1
            
    clusters = list(clusters_dict.values())

    return clusters