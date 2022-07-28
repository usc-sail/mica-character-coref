# author : Sabyasachee

# standard library
import os
import re

# third party
import jsonlines
import nltk
import numpy
import pandas
import spacy
import tqdm

def convert_screenplay_and_coreference_annotation_to_json(data_folder, results_folder):
    '''
    This function converts coreference annotations and screenplays to json objects for further processing.
    Each json object contains the following:

        `rater`       = name of the student worker who annotated the coreference \\
        `movie`       = name of the movie \\
        `token`       = list of tokens \\
        `pos`         = list of part-of-speech tags \\
        `ne`          = list of named entity tags \\
        `parse`       = list of movieparser tags \\
        `sentid`      = list of sentence ids \\
        `begin`       = list of starting token index of mentions \\
        `end`         = list of ending token index of mentions \\
        `character`   = list of character names of mentions

    `token`, `pos`, `ne`, `parse`, and `sentid` are of equal length \\
    `begin`, `end`, and `character` are of equal length
    '''

    #####################################################################
    #### read movie names, raters, and parse
    #####################################################################
    
    parse_filepath = os.path.join(data_folder, "parse.csv")
    movie_filepath = os.path.join(data_folder, "movies.txt")

    parse_df = pandas.read_csv(parse_filepath, index_col=None)
    movies, raters = [], []

    with open(movie_filepath) as fr:
        for line in fr:
            movie, rater = line.split()
            movies.append(movie)
            raters.append(rater)
    
    #####################################################################
    #### initialize spacy model
    #####################################################################
    
    nlp = spacy.load("en_core_web_sm")

    #####################################################################
    #### initialize movie json
    #####################################################################
    
    movie_data = []

    #####################################################################
    #### loop over each movie
    #####################################################################
    
    tbar = tqdm.tqdm(zip(movies, raters), unit="movie", total=len(movies))

    for movie, rater in tbar:

        tbar.set_description(movie)

        #####################################################################
        #### read movie script
        #####################################################################
        
        script_filepath = os.path.join(data_folder, f"screenplay/{movie}.txt")

        with open(script_filepath) as fr:
            script = fr.read()
        lines = script.splitlines()

        #####################################################################
        #### get movie parse
        #####################################################################
        
        tags = parse_df[parse_df["movie"] == movie]["robust"].tolist()
        
        #####################################################################
        #### read coreference annotations
        #####################################################################
        
        annotation_filepath = os.path.join(data_folder, f"csv/{movie}.csv")
        annotation_df = pandas.read_csv(annotation_filepath, index_col=None)

        items = []
        for _, row in annotation_df.iterrows():
            begin, end, character = row["begin"], row["end"], row["entityLabel"]
            items.append((begin, end, character))

        items = sorted(items)

        #####################################################################
        #### find non-whitespace offset of coreference annotations
        #####################################################################
        
        begins, ends, characters, wsbegins, wsends = [], [], [], [], []

        for begin, end, character in items:
            wsbegin = len(re.sub("\s+", "", script[:begin]))
            wsend = wsbegin + len(re.sub("\s+", "", script[begin: end]))
            begins.append(begin)
            ends.append(end)
            characters.append(character)
            wsbegins.append(wsbegin)
            wsends.append(wsend)

        #####################################################################
        #### find segments (blocks of adjacent lines with same movieparser tags)
        #####################################################################
        
        i, c, s = 0, 0, 0
        tokens, tkbegins, tkends, tktags, tksentids, segments, segment_tags = [], [], [], [], [], [], []
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

        #####################################################################
        #### run each segment through spacy pipeline
        #####################################################################
        
        docs = nlp.pipe(segments, batch_size=10200)
        
        #####################################################################
        #### nltk.wordpunct tokenize each spacy token
        #### find tokens, token sentence ids, and token movieparser tags
        #####################################################################
        
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
        
        #####################################################################
        #### match mentions to tokens
        #####################################################################
        
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
        
        #####################################################################
        #### find unmatched mentions and print
        #####################################################################
        
        n_unmatched_begin_indices = sum(i is None for i in matchbegins)
        n_unmatched_end_indices = sum(i is None for i in matchends)

        if n_unmatched_begin_indices:
            print(f"{movie:20s} {n_unmatched_begin_indices:2d} mention begin indexes unmatched")

        if n_unmatched_end_indices:
            print(f"{movie:20s} {n_unmatched_end_indices:2d} mention begin indexes unmatched")
        
        #####################################################################
        #### create speakers array
        #####################################################################
        
        speakers = numpy.full(len(tokens), fill_value="-", dtype=object)

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
        
        #####################################################################
        #### create movie json
        #####################################################################
        
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
    
    #####################################################################
    #### remove tokens with movieparse tag = "C"
    #####################################################################
    
    movie_without_characters_data = remove_characters(movie_data)

    #####################################################################
    #### insert 'says' after tokens with movieparse tag = "C"
    #####################################################################
    
    movie_with_says_data = add_says(movie_data)

    #####################################################################
    #### write movie json
    #####################################################################
    
    filepath = os.path.join(results_folder, "input/normal/movie_coref.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_data:
            writer.write(movie_coref)

    filepath = os.path.join(results_folder, "input/without_characters/movie_without_characters_coref.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_without_characters_data:
            writer.write(movie_coref)

    filepath = os.path.join(results_folder, "input/with_says/movie_with_says_coref.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_with_says_data:
            writer.write(movie_coref)

    #####################################################################
    #### write movie conll
    #####################################################################
    
    movie_conll = convert_movie_coref_json_to_conll(movie_data)
    movie_conll_filepath = os.path.join(results_folder, "input/normal/movie_coref.conll")
    movie_conll.to_csv(movie_conll_filepath, sep="\t", index=False)

    movie_conll = convert_movie_coref_json_to_conll(movie_without_characters_data)
    movie_conll_filepath = os.path.join(results_folder, "input/without_characters/movie_without_characters_coref.conll")
    movie_conll.to_csv(movie_conll_filepath, sep="\t", index=False)

    movie_conll = convert_movie_coref_json_to_conll(movie_with_says_data)
    movie_conll_filepath = os.path.join(results_folder, "input/with_says/movie_with_says_coref.conll")
    movie_conll.to_csv(movie_conll_filepath, sep="\t", index=False)

    #####################################################################
    #### write movie json for wl-coref
    #####################################################################

    movie_wl_data = prepare_for_wlcoref(movie_data)
    filepath = os.path.join(results_folder, "input/normal/movie_coref.wl.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_wl_data:
            writer.write(movie_coref)
    
    movie_wl_data = prepare_for_wlcoref(movie_without_characters_data)
    filepath = os.path.join(results_folder, "input/without_characters/movie_without_characters_coref.wl.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_wl_data:
            writer.write(movie_coref)

    movie_wl_data = prepare_for_wlcoref(movie_with_says_data)
    filepath = os.path.join(results_folder, "input/with_says/movie_with_says_coref.wl.jsonlines")
    with jsonlines.open(filepath, "w") as writer:
        for movie_coref in movie_wl_data:
            writer.write(movie_coref)

def remove_characters(movie_data):
    '''
    This function removes the character names and modifies the mention indexes
    '''
    
    #####################################################################
    #### initialize movie coref
    #####################################################################
    
    movie_coref = []

    #####################################################################
    #### loop over movies
    #####################################################################
    
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")

    for mdata in tbar:

        movie = mdata["movie"]
        tbar.set_description(movie)

        #####################################################################
        #### read rater, tokens, tags, sentids, begins, ends, and characters
        #####################################################################
        
        rater, tokens, tags, sentids, speakers, begins, ends, characters = mdata["rater"], mdata["token"], mdata["parse"], mdata["sentid"], mdata["speaker"], mdata["begin"], mdata["end"], mdata["character"]
        postags, nertags = mdata["pos"], mdata["ne"]
        
        #####################################################################
        #### initalize removed and speakers arrays
        #####################################################################
        
        removed = numpy.zeros(len(tokens), dtype=int)
        # if tags[i: j] == "C" and is followed by some utterance, then we should remove tokens[i: j]
        # removed[: i] remains unchanged
        # removed[i: j] = -1
        # removed[j:] -= j - i

        #####################################################################
        #### find removed array
        #####################################################################
        
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
        
        #####################################################################
        #### initialize the new tokens, tags, sentids, mention indexes, and characters
        #####################################################################
        
        newtokens, newtags, newsentids, newspeakers, newbegins, newends, newcharacters = [], [], [], [], [], [], []
        newpostags, newnertags = [], []

        #####################################################################
        #### get the new tokens, tags, sentids, and speakers
        #####################################################################
        
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
        
        #####################################################################
        #### process sent ids so that adjacent sent ids differ by atmost one
        #####################################################################
        
        i, s = 0, 0
        
        while i < len(newsentids):
            
            j = i + 1
            while j < len(newsentids) and newsentids[j] == newsentids[i]:
                j += 1
            
            for k in range(i, j):
                newsentids[k] = s
            
            s += 1
            i = j

        #####################################################################
        #### modify mention indexes a/c removed
        #### assert that if mention contains any tokens that needs to be
        #### removed, then all mention tags are C
        #####################################################################
        
        for begin, end, character in zip(begins, ends, characters):
            if all(removed[i] != -1 for i in range(begin, end + 1)):
                newbegin = begin - removed[begin]
                newend = end - removed[end]
                newbegins.append(newbegin)
                newends.append(newend)
                newcharacters.append(character)
            else:
                assert all(tags[i] == "C" for i in range(begin, end + 1))
        
        #####################################################################
        #### create movie json
        #####################################################################
        
        movie_coref.append({
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
    
    #####################################################################
    #### return movie coref
    #####################################################################
    
    return movie_coref

def add_says(movie_data):
    '''
    This function inserts 'says' between character name and utterance block. \\
    The function gives the token 'says' a unique tag `A`. \\
    The function also equalizes the sentence ids of the character, 'says', and the first sentence of the
    immediately succeeding utterance block (`D` or `E`).
    '''

    #####################################################################
    #### initialize movie coref
    #####################################################################
    
    movie_coref = []

    #####################################################################
    #### loop over each movie
    #####################################################################
    
    tbar = tqdm.tqdm(movie_data, total=len(movie_data), unit="movie")

    for mdata in tbar:

        movie = mdata["movie"]
        tbar.set_description(movie)

        #####################################################################
        #### read rater, tokens, tags, sentids, mentions, and characters
        #####################################################################
        
        rater, tokens, tags, sentids, speakers, begins, ends, characters = mdata["rater"], mdata["token"], mdata["parse"], mdata["sentid"], mdata["speaker"], mdata["begin"], mdata["end"], mdata["character"]
        postags, nertags = mdata["pos"], mdata["ne"]
        
        #####################################################################
        #### initialize added array
        #####################################################################
        
        added = numpy.zeros(len(tokens), dtype=int)

        #####################################################################
        #### populate added array
        #####################################################################
        
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

        #####################################################################
        #### initialize new tokens, tags, sentids, begins, ends array
        #####################################################################
        
        newtokens, newtags, newsentids, newspeakers, newbegins, newends = [], [], [], [], [], []
        newpostags, newnertags = [], []

        #####################################################################
        #### add 'says' in newtokens
        #####################################################################
        
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

        #####################################################################
        #### equalize sentence id of character, 'says' and first utterance sentence
        #####################################################################
        
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
        
        #####################################################################
        #### process sent ids so that adjacent sent ids differ by atmost once
        #####################################################################
        
        i, s = 0, 0

        while i < len(newsentids):
            j = i + 1
            
            while j < len(newsentids) and newsentids[j] == newsentids[i]:
                j += 1
            
            for k in range(i, j):
                newsentids[k] = s
            
            s += 1
            i = j

        #####################################################################
        #### modify mention indexes a/c added
        #####################################################################
        
        for begin, end in zip(begins, ends):
            newbegin = begin + added[begin]
            newend = end + added[end]
            newbegins.append(newbegin)
            newends.append(newend)
        
        #####################################################################
        #### create movie json
        #####################################################################
        
        movie_coref.append({
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
    
    #####################################################################
    #### return movie json
    #####################################################################
    
    return movie_coref

def convert_movie_coref_json_to_conll(movie_data) -> pandas.DataFrame:
    '''
    Convert movie json coreference to conll format dataframe. \\
    The dataframe contains the following columns:

        - rater
        - movie
        - token
        - sentence_id
        - parse
        - pos
        - ne
        - speaker
        - character_coreference

    Sentences are separated by 1 newline.
    Movies are separated by 2 newlines
    '''

    #####################################################################
    #### initialize dataframe columns
    #####################################################################
    
    rater_col, movie_col, token_col, tag_col, sentid_col, character_col, speaker_col = [], [], [], [], [], [], []
    postag_col, nertag_col = [], []

    #####################################################################
    #### loop over movies
    #####################################################################
    
    for mdata in movie_data:

        movie, rater, tokens, tags, sentids, speakers, begins, ends, characters = mdata["movie"], mdata["rater"], mdata["token"], mdata["parse"], mdata["sentid"], mdata["speaker"], mdata["begin"], mdata["end"], mdata["character"]
        postags, nertags = mdata["pos"], mdata["ne"]
        start = len(rater_col)

        #####################################################################
        #### populate columns
        #####################################################################
        
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

        #####################################################################
        #### populate coreference column
        #####################################################################
        
        for begin, end, character in zip(begins, ends, characters):
            for i in range(begin, end + 1):
                character_col[start + i].append(character)
        
        for i in range(len(tokens)):
            if character_col[start + i]:
                character_col[start + i] = ",".join(sorted(character_col[start + i]))
            else:
                character_col[start + i] = "-"

    #####################################################################
    #### create conll dataframe
    #####################################################################
    
    records = []
    i = 0

    while i < len(rater_col):
        if i and movie_col[i] != movie_col[i - 1]:
            records.append(["" for _ in range(7)])
            records.append(["" for _ in range(7)])
        elif i and sentid_col[i] != sentid_col[i - 1]:
            records.append(["" for _ in range(7)])
        records.append([rater_col[i], movie_col[i], token_col[i], sentid_col[i], postag_col[i], nertag_col[i], tag_col[i], speaker_col[i], character_col[i]])
        i += 1
    
    movie_coref_df = pandas.DataFrame(records, columns=["rater", "movie", "token", "sentence_id", "pos", "named_entity", "parse", "speaker", "character_coreference"])
    
    #####################################################################
    #### return conll dataframe
    #####################################################################
    
    return movie_coref_df

def prepare_for_wlcoref(movie_data):

    movie_coref = []

    for mdata in movie_data:
        movie_coref.append({
            "document_id": f"wb/{mdata['movie']}",
            "cased_words": mdata["token"],
            "sent_id": mdata["sentid"],
            "speaker": mdata["speaker"]
        })
    
    return movie_coref

if __name__=="__main__":
    convert_screenplay_and_coreference_annotation_to_json("/workspace/mica-text-coref/data/annotation", "/workspace/mica-text-coref/results")