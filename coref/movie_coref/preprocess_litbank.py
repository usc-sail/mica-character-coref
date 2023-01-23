"""Preprocess LitBank"""
import spacy
import os
import re
import tqdm
import jsonlines

nlp = spacy.load("en_core_web_lg")
lrec_dir = os.path.join(os.getenv("DATA_DIR"), "lrec2020-coref")
litbank_dir = os.path.join(os.getenv("DATA_DIR"), "litbank")
processed_file = os.path.join(os.getenv("DATA_DIR"), "mica_text_coref/litbank/books.jsonlines")

litbank_data = []

for file in tqdm.tqdm(os.listdir(os.path.join(lrec_dir, "data/original/conll"))):
    if file.endswith(".conll"):
        book_name = file[:-len("_brat.conll")]
        rater = "bamman"
        conll_file = os.path.join(lrec_dir, "data/original/conll", file)
        
        with open(conll_file) as f:
            content = f.read()
        
        lines = content.split("\n")
        words = []
        sent_offsets = []
        starting_coref_ids, ending_coref_ids = [], []
        n_clusters = -1
        
        for line in lines:
            if line.startswith("#begin") or line.startswith("#end"):
                continue
            elif not line.strip():
                if sent_offsets:
                    if sent_offsets[-1][1] < len(words) - 1:
                        sent_offsets.append([sent_offsets[-1][1] + 1, len(words) - 1])
                else:
                    sent_offsets.append([0, len(words) - 1])
            else:
                word = line.split()[3]
                coref = line.split()[-1]
                starting_coref_ids_, ending_coref_ids_ = [], []
                if coref != "_":
                    for match in re.finditer(r"\((\d+)", coref):
                        starting_coref_ids_.append(int(match.group(1)))
                    for match in re.finditer(r"(\d+)\)", coref):
                        ending_coref_ids_.append(int(match.group(1)))
                starting_coref_ids.append(starting_coref_ids_)
                ending_coref_ids.append(ending_coref_ids_)
                words.append(word)
                n_clusters = max([n_clusters] + starting_coref_ids_ + ending_coref_ids_)
        n_clusters += 1

        clusters = []
        for i in range(n_clusters):
            cluster = []
            j = 0
            while j < len(words):
                if i in starting_coref_ids[j]:
                    u = j
                    while j < len(words) and i not in ending_coref_ids[j]:
                            j += 1
                    v = j
                    cluster.append((u, v + 1))
                j += 1
            clusters.append(cluster)
        
        texts = []
        for i, j in sent_offsets:
            sentence = words[i: j + 1]
            text = " ".join(sentence)
            texts.append(text)
        docs = list(nlp.pipe(texts, batch_size=64))

        entity_file = os.path.join(litbank_dir, f"entities/tsv/{book_name}_brat.tsv")
        with open(entity_file) as f:
            entities = [line.split()[1] for line in f if line.strip()]
        ner = []
        for ent in entities:
            if ent == "O":
                ner.append("-")
            elif ent.split("-")[1] == "FAC":
                ner.append("FAC")
            elif ent.split("-")[1] == "PER":
                ner.append("PERSON")
            elif ent.split("-")[1] == "GPE":
                ner.append("GPE")
            elif ent.split("-")[1] == "LOC":
                ner.append("LOC")
            elif ent.split("-")[1] == "ORG":
                ner.append("ORG")
            elif ent.split("-")[1] == "VEH":
                ner.append("VEH")
        
        parse = ["N" for _ in range(len(words))]
        speaker = ["-" for _ in range(len(words))]

        litbank_data.append({"movie": book_name,
                             "rater": rater,
                             "token": words,
                             "ner": ner,
                             "parse": parse,
                             "speaker": speaker,
                             "sent_offset": sent_offsets,
                             "clusters": clusters,
                             "docs": docs})

for book in tqdm.tqdm(litbank_data):
    words = book["token"]
    heads = []
    pos = []
    n = 0
    for (i, j), doc in zip(book["sent_offset"], book["docs"]):
        sentence = words[i: j + 1]
        k = 0
        spacy_word_id_to_litbank_word_id = {}
        for l, word in enumerate(sentence):
            word_ = doc[k].text
            while k < len(doc) and word != word_:
                spacy_word_id_to_litbank_word_id[k] = l
                k += 1
                word_ += doc[k].text
            spacy_word_id_to_litbank_word_id[k] = l
            k += 1
        head_ids = []
        for l, word in enumerate(sentence):
            word_ids_ = [k for k, l_ in spacy_word_id_to_litbank_word_id.items() if l_ == l]
            head_ids_ = [doc[k].head.i for k in word_ids_]
            head_id = max([spacy_word_id_to_litbank_word_id[h] for h in head_ids_])
            head_ids.append(head_id)
            pos_ = [doc[k].tag_ for k in word_ids_]
            pos.append(pos_[-1])
        heads.extend([n + h for h in head_ids])
        n += len(sentence)
    clusters_ = {}
    for c, cluster in enumerate(book["clusters"]):
        cluster_ = []
        for i, j in cluster:
            token_ids_with_outside_head = [k for k in range(i, j) if heads[k] == k or heads[k] < i or heads[k] >= j]
            if len(token_ids_with_outside_head) == 1:
                head = token_ids_with_outside_head[0]
            else:
                head = j - 1
            cluster_.append([i, j - 1, head])
        clusters_[f"ENT_{c + 1}"] = cluster_
    book["clusters"] = clusters_
    book["pos"] = pos
    book.pop("docs")

with jsonlines.open(processed_file, "w") as writer:
    for book in litbank_data:
        writer.write(book)