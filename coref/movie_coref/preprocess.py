"""Functions to convert CSV annotations into JSONLINES format
"""

import os
import jsonlines
import nltk
import numpy as np
import pandas as pd
import re
import spacy
import tqdm

def convert_screenplay_and_coreference_annotation_to_json(
    screenplay_parse_file: str, movie_and_raters_file: str,
    screenplays_dir: str, annotations_dir: str, output_dir: str,
    spacy_model = "en_core_web_sm"):
    '''
    Converts coreference annotations, text screenplays, and
    screenplay parsing tags to json objects for further processing. Each json
    object contains the following:

        `rater`       = name of the student worker who annotated the coreference
        `movie`       = name of the movie
        `token`       = list of tokens
        `pos`         = list of part-of-speech tags
        `ne`          = list of named entity tags
        `parse`       = list of movieparser tags
        `sentid`      = list of sentence ids
        `begin`       = list of starting token index of mentions
        `end`         = list of ending token index of mentions
        `character`   = list of character names of mentions

    `token`, `pos`, `ne`, `parse`, and `sentid` are of equal length. `begin`,
     `end`, and `character` are of equal length. Each json object is further
     converted to conll format and a different jsonlines format used as
    input to train/infer word-level coreference model.

    The function creates three sets of file, each set containing the json
    object, conll, and word-level coreference model json object. The first
    set is saved to `output_dir/regular` and the
    format of the json object is as described above.

    The second set removes character names (and possible corresponding
     annotated mentions) that precede an utterance. The second set is saved to
    `output_dir/nocharacters`.

    The third set adds the word `says` between character names and their
    utterances. It is saved to `output_dir/addsays`.

    Args:
        screenplay_parse_file: csv file containing screenplay parsing tags.
        movie_and_raters_file: text file containing movie and corresponding
            raters names.
        screenplays_dir: directory containing screenplay text files for each
            movie.
        annotations_dir: directory containing csv files of coreference
            annotations.
        output_dir: directory to which the output (json object, conll, and
            json object formatted for word-level coref model, for all three
            sets) is saved.
        spacy_model: name of the english spacy_model
    '''
    # Initialize spacy model and movie data jsonlines
    nlp = spacy.load(spacy_model)
    movie_data = []

    # Read screenplay parse and movie names
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

        # Read movie script, movie parse, and movie coreference annotations
        script_filepath = os.path.join(screenplays_dir, f"{movie}.txt")
        annotation_filepath = os.path.join(annotations_dir, f"{movie}.csv")
        with open(script_filepath, encoding="utf-8") as fr:
            script = fr.read()
        lines = script.splitlines()
        tags = parse_df[parse_df["movie"] == movie]["robust"].tolist()
        annotation_df = pd.read_csv(annotation_filepath, index_col=None)
        items = []
        for _, row in annotation_df.iterrows():
            begin, end, character = row["begin"], row["end"], row["entityLabel"]
            items.append((begin, end, character))
        items = sorted(items)

        # Find non-whitespace offset of coreference annotations
        begins, ends, characters, wsbegins, wsends = [], [], [], [], []
        for begin, end, character in items:
            wsbegin = len(re.sub("\s+", "", script[:begin]))
            wsend = wsbegin + len(re.sub("\s+", "", script[begin: end]))
            begins.append(begin)
            ends.append(end)
            characters.append(character)
            wsbegins.append(wsbegin)
            wsends.append(wsend)

        # Find segments (blocks of adjacent lines with same movieparser tags)
        i, c, s = 0, 0, 0
        tokens, tkbegins, tkends, tktags, tksentids, segments, segment_tags = (
            [], [], [], [], [], [], [])
        postags, nertags = [], []
        while i < len(lines):
            j = i + 1
            while j < len(lines) and tags[j] == tags[i]:
                j += 1
            segment = re.sub("\s+", " ", "\n".join(lines[i: j]).strip())
            if segment:
                segments.append(segment)
                segment_tags.append(tags[i])
            i = j

        # Run each segment through spacy pipeline
        docs = nlp.pipe(segments, batch_size=10200)

        # Tokenize each spacy token using nltk.wordpunct_tokenizer
        # Find tokens, token sentence ids, and token movieparser tags
        for i, doc in enumerate(docs):
            for sent in doc.sents:
                for stoken in sent:
                    text = stoken.text
                    postag = stoken.tag_
                    nertag = stoken.ent_type_
                    if not nertag:
                        nertag = "-"
                    for token in nltk.wordpunct_tokenize(text):
                        tkbegin = c
                        c += len(re.sub("\s+", "", token))
                        tkend = c
                        tksentid = s
                        tokens.append(token)
                        tkbegins.append(tkbegin)
                        tkends.append(tkend)
                        tksentids.append(tksentid)
                        tktags.append(segment_tags[i])
                        postags.append(postag)
                        nertags.append(nertag)
                s += 1

        # Match mentions to tokens
        matchbegins, matchends = [], []
        for begin in wsbegins:
            try:
                i = tkbegins.index(begin)
            except Exception:
                i = None
            matchbegins.append(i)

        for begin, end, wsbegin, wsend in zip(begins, ends, wsbegins, wsends):
            try:
                i = tkends.index(wsend)
            except Exception:
                mention = script[begin: end].rstrip()
                right_context = script[end:].lstrip()
                if mention.endswith(".") and right_context.startswith(".."):
                    wsend -= 1
                    try:
                        i = tkends.index(wsend)
                    except Exception:
                        i = None
                else:
                    i = None
            matchends.append(i)

        # Find unmatched mentions and print
        n_unmatched_begin_indices = sum(i is None for i in matchbegins)
        n_unmatched_end_indices = sum(i is None for i in matchends)
        if n_unmatched_begin_indices:
            print(f"{movie:20s} {n_unmatched_begin_indices:2d} mention begin"
                   " indexes unmatched")
        if n_unmatched_end_indices:
            print(f"{movie:20s} {n_unmatched_end_indices:2d} mention begin"
                   " indexes unmatched")

        # Create speakers array
        speakers = np.full(len(tokens), fill_value="-", dtype=object)
        i = 0
        while i < len(tokens):
            if tktags[i] == "C":
                j = i + 1
                while j < len(tokens) and tktags[j] == tktags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and tktags[k] not in "SC":
                    if tktags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    speaker = " ".join(tokens[i: j])
                    cleaned_speaker = re.sub("\([^\)]+\)", "", speaker).strip()
                    speaker = cleaned_speaker if cleaned_speaker else speaker
                    for l in utterance_token_indices:
                        speakers[l] = speaker
                i = k
            else:
                i += 1
        speakers = speakers.tolist()

        # Create movie json
        movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": tokens,
            "pos": postags,
            "ne": nertags,
            "parse": tktags,
            "sentid": tksentids,
            "speaker": speakers,
            "begin": matchbegins,
            "end": matchends,
            "character": characters,
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
        format_dir = os.path.join(output_dir, input_format)
        os.makedirs(format_dir, exist_ok=True)
        jsonlines_file = os.path.join(format_dir, "movie.jsonlines")
        conll_file = os.path.join(format_dir, "movie.conll")
        wl_jsonlines_file = os.path.join(format_dir, "movie_wl.jsonlines")

        conll = convert_movie_coref_json_to_conll(data)
        wl_data = prepare_for_wlcoref(data)

        with jsonlines.open(jsonlines_file, "w") as writer:
            for d in data:
                writer.write(d)
        conll.to_csv(conll_file, sep="\t", index=False)
        with jsonlines.open(wl_jsonlines_file, "w") as writer:
            for d in wl_data:
                writer.write(d)

def remove_characters(movie_data: list[dict[str, any]]) -> list[dict[str, any]]:
    '''
    Removes character names preceding an utterance and modifies the mention
    indexes.
    '''
    # Initialize new movie data
    new_movie_data = []

    # Loop over movies
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")
    for mdata in tbar:
        movie = mdata["movie"]
        tbar.set_description(movie)
        (rater, tokens, tags, sentids, speakers, begins, ends, characters) = (
            mdata["rater"], mdata["token"], mdata["parse"], mdata["sentid"],
            mdata["speaker"], mdata["begin"], mdata["end"], mdata["character"])
        postags, nertags = mdata["pos"], mdata["ne"]
        removed = np.zeros(len(tokens), dtype=int)

        # if tags[i: j] == "C" and is followed by some utterance, 
        # then we should remove tokens[i: j]
        # removed[: i] remains unchanged
        # removed[i: j] = -1
        # removed[j:] -= j - i
        i = 0
        while i < len(tokens):
            if tags[i] == "C":
                j = i + 1
                while j < len(tokens) and tags[j] == tags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and tags[k] not in "SC":
                    if tags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    removed[i: j] = -1
                    removed[j:] += j - i
                i = k
            else:
                i += 1
        removed = removed.tolist()

        # Find the new tokens, tags, sentence ids, pos tags, and ner tags
        (newtokens, newtags, newsentids, newspeakers, newbegins, newends, 
         newcharacters) = [], [], [], [], [], [], []
        newpostags, newnertags = [], []
        i = 0
        while i < len(tokens):
            if removed[i] != -1:
                newtokens.append(tokens[i])
                newtags.append(tags[i])
                newspeakers.append(speakers[i])
                newsentids.append(sentids[i])
                newpostags.append(postags[i])
                newnertags.append(nertags[i])
            i += 1

        # Process sent ids so that adjacent sent ids differ by atmost one
        i, s = 0, 0
        while i < len(newsentids):
            j = i + 1
            while j < len(newsentids) and newsentids[j] == newsentids[i]:
                j += 1
            for k in range(i, j):
                newsentids[k] = s
            s += 1
            i = j

        # Modify mention indexes a/c removed
        # Assert that if mention contains any tokens that needs to be
        # removed, then all mention tags are C
        for begin, end, character in zip(begins, ends, characters):
            if all(removed[i] != -1 for i in range(begin, end + 1)):
                newbegin = begin - removed[begin]
                newend = end - removed[end]
                newbegins.append(newbegin)
                newends.append(newend)
                newcharacters.append(character)
            else:
                assert all(tags[i] == "C" for i in range(begin, end + 1))

        # Create movie json
        new_movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": newtokens,
            "parse": newtags,
            "pos": newpostags,
            "ne": newnertags,
            "sentid": newsentids,
            "speaker": newspeakers,
            "begin": newbegins,
            "end": newends,
            "character": newcharacters
        })

    return new_movie_data

def add_says(movie_data: list[dict[str, any]]) -> list[dict[str, any]]:
    '''
    Inserts 'says' between character name and utterance block. Give the token
    'says' a unique tag `A`. The function also equalizes the sentence ids of
    the character, 'says', and the first sentence of the immediately succeeding
    utterance block (`D` or `E`).
    '''
    # Initialize new movie data
    new_movie_data = []

    # Loop over each movie
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")
    for mdata in tbar:
        movie = mdata["movie"]
        tbar.set_description(movie)
        rater, tokens, tags, sentids, speakers, begins, ends, characters = (
            mdata["rater"], mdata["token"], mdata["parse"], mdata["sentid"],
            mdata["speaker"], mdata["begin"], mdata["end"], mdata["character"])
        postags, nertags = mdata["pos"], mdata["ne"]

        added = np.zeros(len(tokens), dtype=int)
        i = 0
        while i < len(tokens):
            if tags[i] == "C":
                j = i + 1
                while j < len(tokens) and tags[j] == tags[i]:
                    j += 1
                k = j
                utterance_token_indices = []
                while k < len(tokens) and tags[k] not in "SC":
                    if tags[k] in "DE":
                        utterance_token_indices.append(k)
                    k += 1
                if utterance_token_indices:
                    added[j - 1] = -1
                    added[j:] += 1
                i = k
            else:
                i += 1
        added = added.tolist()

        # Find new tokens, tags, sentids, begins, ends array
        newtokens, newtags, newsentids, newspeakers, newbegins, newends = (
            [], [], [], [], [], [])
        newpostags, newnertags = [], []

        # Add 'says' in newtokens
        i = 0
        while i < len(tokens):
            newtokens.append(tokens[i])
            newtags.append(tags[i])
            newsentids.append(sentids[i])
            newspeakers.append(speakers[i])
            newpostags.append(postags[i])
            newnertags.append(nertags[i])
            if added[i] == -1:
                newtokens.append("says")
                newtags.append("A")
                newsentids.append(sentids[i])
                newspeakers.append("-")
                newpostags.append("VBZ")
                newnertags.append("-")
            i += 1

        # Equalize sentence id of character, 'says' and first utterance sentence
        i = 0
        while i < len(newtokens):
            if newtags[i] == "A" and i < len(newtokens) - 1 and newtags[i + 1] in "DE":
                j = i + 1
                sentid = newsentids[j]
                while j < len(newtokens) and newtags[j] in "DE" and newsentids[j] == sentid:
                    newsentids[j] = newsentids[i]
                    j += 1
                i = j
            else:
                i += 1

        # Process sent ids so that adjacent sent ids differ by atmost once
        i, s = 0, 0
        while i < len(newsentids):
            j = i + 1
            while j < len(newsentids) and newsentids[j] == newsentids[i]:
                j += 1
            for k in range(i, j):
                newsentids[k] = s
            s += 1
            i = j

        # Modify mention indexes a/c added
        for begin, end in zip(begins, ends):
            newbegin = begin + added[begin]
            newend = end + added[end]
            newbegins.append(newbegin)
            newends.append(newend)

        # Create the new movie json
        new_movie_data.append({
            "movie": movie,
            "rater": rater,
            "token": newtokens,
            "pos": newpostags,
            "ne": newnertags,
            "parse": newtags,
            "sentid": newsentids,
            "speaker": newspeakers,
            "begin": newbegins,
            "end": newends,
            "character": characters
        })

    return new_movie_data

def convert_movie_coref_json_to_conll(movie_data: list[dict[str, any]]) -> (
    pd.DataFrame):
    '''
    Convert movie json coreference to conll format dataframe. The dataframe
    contains the following columns:

        - rater
        - movie
        - token
        - sentence_id
        - parse
        - pos
        - ne
        - speaker
        - character_coreference

    Sentences are separated by 1 newline. Movies are separated by 2 newlines.
    '''
    # Initialize dataframe columns
    (rater_col, movie_col, token_col, tag_col, sentid_col, character_col,
     speaker_col) = [], [], [], [], [], [], []
    postag_col, nertag_col = [], []

    # Loop over movies
    for mdata in movie_data:
        (movie, rater, tokens, tags, sentids, speakers, begins, ends,
         characters) = (mdata["movie"], mdata["rater"], mdata["token"],
                        mdata["parse"], mdata["sentid"], mdata["speaker"],
                        mdata["begin"], mdata["end"], mdata["character"])
        postags, nertags = mdata["pos"], mdata["ne"]
        start = len(rater_col)

        # Populate columns
        for i in range(len(tokens)):
            rater_col.append(rater)
            movie_col.append(movie)
            token_col.append(tokens[i])
            tag_col.append(tags[i])
            sentid_col.append(sentids[i])
            character_col.append([])
            speaker_col.append(speakers[i])
            postag_col.append(postags[i])
            nertag_col.append(nertags[i])

        for begin, end, character in zip(begins, ends, characters):
            for i in range(begin, end + 1):
                character_col[start + i].append(character)

        for i in range(len(tokens)):
            if character_col[start + i]:
                character_col[start + i] = ",".join(sorted(character_col[start + i]))
            else:
                character_col[start + i] = "-"

    # Create conll dataframe
    records = []
    i = 0
    while i < len(rater_col):
        if i and movie_col[i] != movie_col[i - 1]:
            records.append(["" for _ in range(7)])
            records.append(["" for _ in range(7)])
        elif i and sentid_col[i] != sentid_col[i - 1]:
            records.append(["" for _ in range(7)])
        records.append([rater_col[i], movie_col[i], token_col[i], sentid_col[i],
                        postag_col[i], nertag_col[i], tag_col[i],
                        speaker_col[i], character_col[i]])
        i += 1

    movie_coref_df = pd.DataFrame(records,
                                  columns=["rater", "movie", "token",
                                           "sentence_id", "pos", "named_entity",
                                           "parse", "speaker",
                                           "character_coreference"])
    return movie_coref_df

def prepare_for_wlcoref(movie_data: list[dict[str, any]]) -> (
    list[dict[str, any]]):
    """Convert to jsonlines format which can be used as input to the word-level
    coreference model.
    """
    new_movie_data = []
    for mdata in movie_data:
        new_movie_data.append({
            "document_id": f"wb/{mdata['movie']}",
            "cased_words": mdata["token"],
            "sent_id": mdata["sentid"],
            "speaker": mdata["speaker"]
        })
    return new_movie_data