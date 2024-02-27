"""Functions to convert CSV annotations into JSONLINES format"""

import os
import jsonlines
import nltk
import numpy as np
import pandas as pd
import re
import spacy
import tqdm
import unidecode
from copy import deepcopy

def preprocess_scripts(script_files: list[str], parse_files: list[str], output_file = None, gpu_device = -1):
    """Preprocess scripts and parse files into json files for coreference resolution."""
    # Initialize spacy model
    if gpu_device >= 0:
        spacy.require_gpu(gpu_id=gpu_device) # type: ignore
    nlp = spacy.load("en_core_web_sm")

    # initialize movie data
    movie_data = []

    # loop over script and parse files
    for script_file, parse_file in tqdm.tqdm(zip(script_files, parse_files), total=len(script_files), unit="script"):

        # read parse file
        with open(parse_file, "r") as fr:
            tags = fr.read().strip().split("\n")
        
        # read script file
        with open(script_file, "r", encoding="utf-8") as fr:
            script_lines = fr.read().strip().split("\n")
        
        # assert number of script lines equal the number of tags
        assert len(script_lines) == len(tags), (f"Number of script lines ({len(script_lines)}) do not equal number of "
                                                f"parse tags ({len(tags)}) for script file = {script_file} and parse "
                                                f"file = {parse_file}")
        
        # Extract segments from script lines (adjacent lines with same movieparser tags)
        i = 0
        segment_texts, segment_tags = [], []
        while i < len(script_lines):
            j = i + 1
            while j < len(script_lines) and tags[j] == tags[i]:
                j += 1
            segment = re.sub(r"\s+", " ", " ".join(script_lines[i: j]).strip())
            segment = (" ".join(nltk.wordpunct_tokenize(segment))).strip()
            segment = re.sub(r"\s+", " ", segment.strip())
            if segment:
                segment_texts.append(segment)
                segment_tags.append(tags[i])
            i = j
        
        # Process each segment through spacy pipeline
        docs = nlp.pipe(segment_texts, batch_size=512)

        # Extract tokens, part-of-speech tag, named entity tag, parse tag, and sentence id of each token
        tokens, token_postags, token_nertags, token_tags, token_sentids, token_dep_headids, token_dep_tags = (
            [], [], [], [], [], [] ,[])
        c, s = 0, 0
        for i, doc in tqdm.tqdm(enumerate(docs), total=len(segment_texts)):
            for sent in doc.sents:
                for stoken in sent:
                    text = stoken.text
                    ascii_text = unidecode.unidecode(text, errors="ignore").strip()
                    postag = stoken.tag_
                    nertag = stoken.ent_type_
                    if not nertag:
                        nertag = "-"
                    token_sentid = s
                    tokens.append(ascii_text)
                    token_sentids.append(token_sentid)
                    token_tags.append(segment_tags[i])
                    token_postags.append(postag)
                    token_nertags.append(nertag)
                    token_dep_headids.append(stoken.head.i + c)
                    token_dep_tags.append(stoken.dep_)
                s += 1
            c += len(doc)

        # Create speakers array
        speakers = np.full(len(tokens), fill_value="-", dtype=object)
        i = 0
        while i < len(tokens):
            if token_tags[i] == "C":
                j = i + 1
                while j < len(tokens) and token_tags[j] == token_tags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and token_tags[k] not in "SC":
                    if token_tags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    speaker = " ".join(tokens[i: j])
                    cleaned_speaker = re.sub(r"\([^\)]+\)", "", speaker).strip()
                    speaker = cleaned_speaker if cleaned_speaker else speaker
                    for l in utterance_token_indices:
                        speakers[l] = speaker
                i = k
            else:
                i += 1
        speakers = speakers.tolist()
        
        # Find sentence offsets
        sentence_offsets: list[list[int]] = []
        i = 0
        while i < len(token_sentids):
            j = i + 1
            while j < len(token_sentids) and token_sentids[i] == token_sentids[j]:
                j += 1
            sentence_offsets.append([i, j - 1])
            i = j

        # Create movie json
        movie_data.append({
            "movie": os.path.basename(script_file),
            "rater": "example",
            "token": tokens,
            "pos": token_postags,
            "ner": token_nertags,
            "parse": token_tags,
            "speaker": speakers,
            "dep_head": token_dep_headids,
            "dep": token_dep_tags,
            "sent_offset": sentence_offsets,
            "clusters": {}
        })
    
    # write jsonlines
    if output_file is not None:
        with jsonlines.open(output_file, "w") as writer:
            for obj in movie_data:
                writer.write(obj) # type: ignore

    # return movie data
    return movie_data

def convert_screenplay_and_coreference_annotation_to_json(screenplay_parse_file: str, 
    movie_and_raters_file: str, screenplays_dir: str, annotations_dir: str, output_dir: str,
    spacy_model = "en_core_web_sm", spacy_gpu_device = -1):
    """
    Converts coreference annotations, text screenplays, and screenplay parsing tags to json objects
    for further processing. Each json object contains the following:

    Attributes:
        rater: Name of the student worker who annotated the coreference
        movie: Name of the movie 
        token: List of tokens 
        pos: List of part-of-speech tags 
        ner: List of named entity tags 
        parse: List of movieparser tags 
        speaker: List of speakers 
        sent_offset: List of sentence offsets. Each sentence offset is a list of two integers:
            start and end. The end is included, therefore you can get the sentence text as
            tokens[start: end]
        clusters : Dictionary of character to list of mentions. Each mention is a list of three
            integers: start, end, and head. The head is the index of the mention's head word,
            therefore start <= head <= end. The end is inclusive, therefore you can get the mention
            text as tokens[start: end + 1]

    token, pos, ner, parse, and speaker are of equal length. Each json object is further converted
    to a different json format used as input to a pre-trained wl-RoBERTa for inference. The function
    creates three sets of file, each set containing the json files. The first set of files is saved
    to output_dir/regular and the format of the json file is as described above. The second set
    removes character names (and possible corresponding annotated mentions) that precede an
    utterance. The second set is saved to output_dir/nocharacters. The third set adds the word
    "says" between character names and their utterances. It is saved to output_dir/addsays.

    Args:
        screenplay_parse_file: csv file containing screenplay parsing tags.
        movie_and_raters_file: text file containing movie and corresponding raters names.
        screenplays_dir: directory containing screenplay text files for each movie.
        annotations_dir: directory containing csv files of coreference annotations.
        output_dir: directory to which the output (json object, conll, and json object formatted
            for word-level coref model, for all three sets) is saved.
        spacy_model: name of the english spacy_model
        spacy_gpu_device: GPU to use for spacy
    """
    # Initialize spacy model
    if spacy_gpu_device >= 0:
        spacy.require_gpu(gpu_id=spacy_gpu_device) # type: ignore
    nlp = spacy.load(spacy_model)
    movie_data = []

    # Read screenplay parse, movie names, and raters
    movies, raters = [], []
    parse_df = pd.read_csv(screenplay_parse_file, index_col=None)
    with open(movie_and_raters_file) as fr:
        for line in fr:
            movie, rater = line.split()
            movies.append(movie)
            raters.append(rater)

    # Loop over each movie
    tbar = tqdm.tqdm(zip(movies, raters), unit="movie", total=len(movies))
    for movie, rater in tbar:
        tbar.set_description(movie)

        # Read movie script and annotations
        script_filepath = os.path.join(screenplays_dir, f"{movie}.txt")
        annotation_filepath = os.path.join(annotations_dir, f"{movie}.csv")
        with open(script_filepath, encoding="utf-8") as fr:
            script = fr.read()
        lines = script.splitlines()
        parsetags = parse_df[parse_df["movie"] == movie]["robust"].tolist()
        annotation_df = pd.read_csv(annotation_filepath, index_col=None)
        items = []
        for _, row in annotation_df.iterrows():
            begin, end, character = row["begin"], row["end"], row["entityLabel"]
            items.append((begin, end, character))
        items = sorted(items)

        # Find non-whitespace offset of annotated mentions
        begins, ends, characters, wsbegins, wsends = [], [], [], [], []
        prev_begin, prev_wsbegin = 0, 0
        for i, (begin, end, character) in enumerate(items):
            if i == 0:
                wsbegin = len(re.sub(r"\s", "", script[:begin]))
            else:
                wsbegin = prev_wsbegin + len(re.sub(r"\s", "", script[prev_begin: begin]))
            prev_begin, prev_wsbegin = begin, wsbegin
            wsend = wsbegin + len(re.sub(r"\s", "", script[begin: end]))
            begins.append(begin)
            ends.append(end)
            characters.append(character)
            wsbegins.append(wsbegin)
            wsends.append(wsend)

        # Extract segments from script lines (adjacent lines with same 
        # movieparser tags)
        i = 0
        segment_texts, segment_tags = [], []
        while i < len(lines):
            j = i + 1
            while j < len(lines) and parsetags[j] == parsetags[i]:
                j += 1
            segment = re.sub(r"\s+", " ", " ".join(lines[i: j]).strip())
            segment = (" ".join(nltk.wordpunct_tokenize(segment))).strip()
            segment = re.sub(r"\s+", " ", segment.strip())
            if segment:
                segment_texts.append(segment)
                segment_tags.append(parsetags[i])
            i = j

        # Process each segment through spacy pipeline
        docs = nlp.pipe(segment_texts, batch_size=10200)

        # Extract tokens, and the head index, part-of-speech tag, named entity
        # tag, offsets, parse tag, and sentence id of each token
        (tokens, token_heads, token_postags, token_nertags, token_begins, token_ends, token_tags,
            token_sentids) = [], [], [], [], [], [], [], []
        c, s, n = 0, 0, 0
        for i, doc in enumerate(docs):
            for sent in doc.sents:
                for stoken in sent:
                    text = stoken.text
                    ascii_text = unidecode.unidecode(text, errors="strict")
                    assert ascii_text != "", f"token='{text}', Empty ascii text"
                    postag = stoken.tag_
                    nertag = stoken.ent_type_
                    if not nertag:
                        nertag = "-"
                    token_begin = c
                    c += len(re.sub(r"\s+", "", text))
                    token_end = c
                    token_sentid = s
                    tokens.append(ascii_text)
                    token_heads.append(n + stoken.head.i)
                    token_begins.append(token_begin)
                    token_ends.append(token_end)
                    token_sentids.append(token_sentid)
                    token_tags.append(segment_tags[i])
                    token_postags.append(postag)
                    token_nertags.append(nertag)
                n += len(sent)
                s += 1

        # Match mention offsets to token offsets
        # Unmatched mentions will printed
        mention_begins, mention_ends, mention_characters = [], [], []
        n_unmatched = 0
        for wsbegin, wsend, begin, end, character in zip(
            wsbegins, wsends, begins, ends, characters):
            try:
                i = token_begins.index(wsbegin)
            except Exception:
                i = None
            try:
                j = token_ends.index(wsend)
            except Exception:
                mention = script[begin: end].rstrip()
                right_context = script[end:].lstrip()
                if mention.endswith(".") and right_context.startswith(".."):
                    wsend -= 1
                    try:
                        j = token_ends.index(wsend)
                    except Exception:
                        j = None
                else:
                    j = None
            if i is None or j is None:
                mention = script[begin: end]
                context = script[begin-10: end+10]
                print(f"mention = '{mention}'")
                print(f"context = '{context}'")
                if i is None:
                    print("Could not token-map start of mention")
                if j is None:
                    print("Could not token-map end of mention")
                print()
            if i is not None and j is not None:
                mention_begins.append(i)
                mention_ends.append(j)
                mention_characters.append(character)
            else:
                n_unmatched += 1
        
        if n_unmatched > 0:
            print(f"{n_unmatched} mentions could not be token-mapped!")
            print("If this number is (>10), annotations might have too many errors")

        # Create speakers array
        speakers = np.full(len(tokens), fill_value="-", dtype=object)
        i = 0
        while i < len(tokens):
            if token_tags[i] == "C":
                j = i + 1
                while j < len(tokens) and token_tags[j] == token_tags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and token_tags[k] not in "SC":
                    if token_tags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    speaker = " ".join(tokens[i: j])
                    cleaned_speaker = re.sub(r"\([^\)]+\)", "", speaker).strip()
                    speaker = cleaned_speaker if cleaned_speaker else speaker
                    for l in utterance_token_indices:
                        speakers[l] = speaker
                i = k
            else:
                i += 1
        speakers = speakers.tolist()

        # Create character to mention offsets and head
        clusters: dict[str, list[list[int]]] = {}
        for character, mention_begin, mention_end in zip(
            mention_characters, mention_begins, mention_ends):
            if character not in clusters:
                clusters[character] = []
            token_indexes_with_outside_head = []
            for i in range(mention_begin, mention_end + 1):
                head_index = token_heads[i]
                if (head_index == i or head_index < mention_begin or
                    head_index > mention_end):
                    token_indexes_with_outside_head.append(i)
            mention_head = mention_end
            if len(token_indexes_with_outside_head) == 1:
                mention_head = token_indexes_with_outside_head[0]
            clusters[character].append([mention_begin, mention_end, mention_head])
        
        # Find sentence offsets
        sentence_offsets: list[list[int]] = []
        i = 0
        while i < len(token_sentids):
            j = i + 1
            while j < len(token_sentids) and token_sentids[i] == token_sentids[j]:
                j += 1
            sentence_offsets.append([i, j - 1])
            i = j

        # Create movie json
        movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": tokens,
            "pos": token_postags,
            "ner": token_nertags,
            "parse": token_tags,
            "speaker": speakers,
            "sent_offset": sentence_offsets,
            "clusters": clusters
        })

    # Remove tokens with movieparse tag = "C"
    movie_nocharacters_data = remove_characters(movie_data)

    # Insert 'says' after tokens with movieparse tag = "C"
    movie_addsays_data = add_says(movie_data)

    # Write movie jsonlines, movie conll, and jsonlines formatted for
    # word-level coreference model
    for input_format, data in [("regular", movie_data), 
                               ("nocharacters", movie_nocharacters_data), 
                               ("addsays", movie_addsays_data)]:
        for partition in ["train", "dev"]:
            for wl in [False, True]:
                format_dir = os.path.join(output_dir, input_format)
                os.makedirs(format_dir, exist_ok=True)
                if partition == "train":
                    write_data = [d for d in data if d["rater"] != "expert"]
                else:
                    write_data = [d for d in data if d["rater"] == "expert"]
                if wl:
                    write_data = prepare_for_wlcoref(write_data)
                    jsonlines_file = os.path.join(
                        format_dir, f"{partition}_wl.jsonlines")
                else:
                    jsonlines_file = os.path.join(
                        format_dir, f"{partition}.jsonlines")

                print(f"writing to {jsonlines_file}")
                with jsonlines.open(jsonlines_file, "w") as writer:
                    for d in write_data:
                        writer.write(d) # type: ignore

def remove_characters(movie_data: list[dict[str, any]]) -> list[dict[str, any]]:
    """Removes character names preceding an utterance.
    """
    # Initialize new movie data
    new_movie_data = []

    # Loop over movies
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")
    for mdata in tbar:
        (movie, rater, tokens, postags, nertags, parsetags, sentence_offsets, speakers, 
            clusters) = (mdata["movie"], mdata["rater"], mdata["token"], mdata["pos"], mdata["ner"],
                mdata["parse"], mdata["sent_offset"], mdata["speaker"], mdata["clusters"])
        tbar.set_description(movie)

        # removed[x] is the number of tokens to remove from tokens[:x]
        # if tags[i: j] == "C" and is followed by some utterance, 
        # then we should remove tokens[i: j]
        # removed[: i] remains unchanged
        # removed[i: j] = -1
        # removed[j:] += j - i
        removed = np.zeros(len(tokens), dtype=int)
        i = 0
        while i < len(tokens):
            if parsetags[i] == "C":
                j = i + 1
                while j < len(tokens) and parsetags[j] == parsetags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and parsetags[k] not in "SC":
                    if parsetags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    removed[i: j] = -1
                    removed[j:] += j - i
                i = k
            else:
                i += 1

        # Skip the tokens marked for removal
        newtokens, newpostags, newnertags, newparsetags, newspeakers = [], [], [], [], []
        i = 0
        while i < len(tokens):
            if removed[i] != -1:
                newtokens.append(tokens[i])
                newpostags.append(postags[i])
                newnertags.append(nertags[i])
                newparsetags.append(parsetags[i])
                newspeakers.append(speakers[i])
            i += 1

        # Find new sentence offsets
        new_sentence_offsets = []
        for i, j in sentence_offsets:
            assert all(removed[i: j + 1] == -1) or all(removed[i: j + 1] != -1), (
                "All tokens or none of the tokens of a sentence should be "
                "removed")
            if all(removed[i: j + 1] != -1):
                i = i - int(removed[i])
                j = j - int(removed[j])
                new_sentence_offsets.append([i, j])

        # Find new clusters
        new_clusters: dict[str, list[list[int]]] = {}
        for character, mentions in clusters.items():
            new_mentions = []
            for begin, end, head in mentions:
                assert (all(removed[begin: end + 1] == -1) or all(removed[begin: end + 1] != -1)), (
                    "All tokens or none of the tokens of a mention should be removed, mention = "
                    f"[{begin},{end},{head}]")
                if all(removed[begin: end + 1] != -1):
                    begin = begin - int(removed[begin])
                    end = end - int(removed[end])
                    head = head - int(removed[head])
                    new_mentions.append([begin, end, head])
            if new_mentions:
                new_clusters[character] = new_mentions

        # Create movie json
        new_movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": newtokens,
            "pos": newpostags,
            "ner": newnertags,
            "parse": newparsetags,
            "speaker": newspeakers,
            "sent_offset": new_sentence_offsets,
            "clusters": new_clusters
        })

    return new_movie_data

def add_says(movie_data: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Inserts 'says' between character name and utterance block. Give the token
    'says' a unique tag `A`.
    """
    # Initialize new movie data
    new_movie_data = []

    # Loop over each movie
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")
    for mdata in tbar:
        (movie, rater, tokens, postags, nertags, parsetags, sentence_offsets, speakers,
            clusters) = (mdata["movie"], mdata["rater"], mdata["token"], mdata["pos"], mdata["ner"],
                mdata["parse"], mdata["sent_offset"], mdata["speaker"], mdata["clusters"])
        tbar.set_description(movie)

        # added[x] is the number of 'says' added in tokens[0...x] or tokens[:x + 1]
        # Therefore the token at position x is mapped to x + added[x]
        added = np.zeros(len(tokens), dtype=int)
        i = 0
        while i < len(tokens):
            if parsetags[i] == "C":
                j = i + 1
                while j < len(tokens) and parsetags[j] == parsetags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and parsetags[k] not in "SC":
                    if parsetags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    added[j:] += 1
                i = k
            else:
                i += 1

        # Find new tokens, pos tags, ner tags, parse tags, and speakers
        newtokens, newpostags, newnertags, newparsetags, newspeakers = (
            [], [], [], [], [])
        i = 0
        while i < len(tokens):
            newtokens.append(tokens[i])
            newpostags.append(postags[i])
            newnertags.append(nertags[i])
            newparsetags.append(parsetags[i])
            newspeakers.append(speakers[i])
            if i < len(tokens) - 1 and added[i] < added[i + 1]:
                newtokens.append("says")
                newpostags.append("VBZ")
                newnertags.append("-")
                newparsetags.append("A")
                newspeakers.append("-")
            i += 1

        # Find new sentence offsets
        new_sentence_offsets = []
        k = 0
        while k < len(sentence_offsets):
            i, j = sentence_offsets[k]
            offset = 0
            if k < len(sentence_offsets) - 1 and added[j] < added[j + 1]:
                if parsetags[j + 1] in "DE":
                    k = k + 1
                    j = sentence_offsets[k][1]
                else:
                    offset = 1
            i = i + int(added[i])
            j = j + int(added[j]) + offset
            k = k + 1
            new_sentence_offsets.append([i, j])

        # Find new clusters
        new_clusters: dict[str, list[list[int]]] = {}
        for character, mentions in clusters.items():
            new_mentions = []
            for begin, end, head in mentions:
                begin = begin + int(added[begin])
                end = end + int(added[end])
                head = head + int(added[head])
                new_mentions.append([begin, end, head])
            new_clusters[character] = new_mentions

        # Create the new movie json
        new_movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": newtokens,
            "pos": newpostags,
            "ner": newnertags,
            "parse": newparsetags,
            "speaker": newspeakers,
            "sent_offset": new_sentence_offsets,
            "clusters": new_clusters
        })

    return new_movie_data

def prepare_for_wlcoref(movie_data: list[dict[str, any]]) -> (
    list[dict[str, any]]):
    """Convert to jsonlines format which can be used as input to the word-level coreference model.
    """
    new_movie_data = []
    for mdata in movie_data:
        parse_ids = []
        i, c = 0, 0
        while i < len(mdata["parse"]):
            j = i + 1
            while j < len(mdata["parse"]) and mdata["parse"][j] == mdata["parse"][i]:
                j += 1
            parse_ids.extend([c] * (j - i))
            c += 1
            i = j
        _mdata = deepcopy(mdata)
        _mdata["document_id"] = "wb_" + mdata["movie"]
        _mdata["part_id"] = 0
        _mdata["cased_words"] = mdata["token"]
        _mdata["sent_id"] = parse_ids

        head2span: set[tuple[int, int, int]] = set()
        word_clusters: list[set[int]] = []
        span_clusters: list[set[tuple[int, int]]] = []
        for _, cluster in _mdata["clusters"].items():
            word_cluster: set[int] = set()
            span_cluster: set[tuple[int, int]] = set()
            for begin, end, head in cluster:
                head2span.add((begin, end + 1, head))
                word_cluster.add(head)
                span_cluster.add((begin, end + 1))
            word_clusters.append(word_cluster)
            span_clusters.append(span_cluster)

        _mdata["head2span"] = [[head, begin, end] for head, begin, end in head2span]
        _mdata["word_clusters"] = [sorted(word_cluster) for word_cluster in word_clusters]
        _mdata["span_clusters"] = [sorted([list(span) for span in span_cluster]) for span_cluster in span_clusters]
        new_movie_data.append(_mdata)
    return new_movie_data