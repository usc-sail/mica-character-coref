import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

script_index = pd.read_csv('/data/sbaruah/movie/dataset/index.csv', index_col = None)
nlp = spacy.load('en_core_web_sm')
pronouns = 'I, me, my, myself, you, your, yourself, she, her, herself, he, him, his, himself, it, its, we, our, they, them, their'.split(', ') + ['thou','thee','thine','ye']

mica_ids = script_index[script_index['text'] & script_index['parsed'] & (script_index['source'] == 'mica')].drop_duplicates('imdb_id')['id'].values
mica_scripts = [open(f'/data/sbaruah/movie/dataset/script_dataset/{id}/{id}_parsed.txt').read().strip().split('\n') for id in tqdm(mica_ids)]

mica_docs_arr = []

for script in tqdm(mica_scripts):
    texts = []
    
    for line in script:
        tag = line[0]
        if tag == 'D' or tag == 'N':
            texts.append(line[3:])
            
    docs = [nlp(text) for text in texts]
    mica_docs_arr.append(docs)

n_persons_arr = []
n_pronouns_arr = []
pronoun_counts_arr = []

for docs in tqdm(mica_docs_arr):
    pronoun_counts = dict()
    n_pronouns = 0
    
    for doc in docs:
        for token in doc:
            if token.text in pronouns:
                if token.text not in pronoun_counts:
                    pronoun_counts[token.text] = 0
                pronoun_counts[token.text] += 1
                n_pronouns += 1
    
    n_pronouns_arr.append(n_pronouns)
    pronoun_counts_arr.append(pronoun_counts)
    
    n_persons = 0
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                n_persons += 1
    
    n_persons_arr.append(n_persons)

plt.figure(figsize = (20, 8))
plt.subplot(1, 2, 1)
plt.hist(n_persons_arr, bins = 20)
plt.xlabel('#PERSONS')
plt.ylabel('#MICA SCRIPTS')
plt.title('Histogram of #PERSONS')

plt.subplot(1, 2, 2)
plt.hist(n_pronouns_arr, bins = 20)
plt.xlabel('#PRONOUNS')
plt.ylabel('#MICA SCRIPTS')
plt.title('Histogram of #PRONOUNS')

plt.savefig('results/pronoun_person_histogram.png')
plt.close()

ratio_arr = np.array(n_pronouns_arr)/np.array(n_persons_arr)
plt.hist(ratio_arr, bins = 50)
plt.xlabel('ratio of pronouns to persons count')
plt.ylabel('#scripts')
plt.title('Histogram of #pronouns:#persons from MICA scripts')
plt.savefig('results/pronoun_person_ratio.png')
plt.close()

pronoun_total_count = dict()
for pronoun in pronouns:
    t = 0
    for pronoun_counts in pronoun_counts_arr:
        if pronoun in pronoun_counts:
            t += pronoun_counts[pronoun]
    pronoun_total_count[pronoun] = t

plt.figure(figsize = (30, 6))
plt.plot(np.arange(len(pronouns)), list(pronoun_total_count.values()))
plt.xticks(np.arange(len(pronouns)), labels=pronoun_total_count.keys())
plt.ylabel('Total Count')
plt.title('Total count of pronouns in MICA scripts')
plt.savefig('results/pronoun_total_count.png')
plt.close()