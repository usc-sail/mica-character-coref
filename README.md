# Character Coreference Resolution in Movie Screenplays

This repository contains the source code and data for training a coreference resolution model for characters in movie
screenplays.

The work has been accepted to ACL Findings 2023.
You can read the paper 
[Character Coreference Resolution in Movie Screenplays](https://aclanthology.org/2023.findings-acl.654/)
to get more details. <br>
This builds upon our prior work in 
[Annotation and Evaluation of Coreference Resolution in Screenplays](https://aclanthology.org/2021.findings-acl.176/).

## Data

The __MovieCoref__ corpus is saved to the [data](data/) directory.

The __MovieCoref__ coreference annotations and screenplay documents can be found in the [data/labels](data/labels/) and [data/screenplay](data/screenplay/) directories, respectively.
[data/movies.txt](data/movies.txt) contains the name of the movies and their raters.
[data/parse.csv](data/parse.csv) contains the line-level structural tags of the screenplays, obtained after screenplay parsing.

The [data/addsays](data/addsays/), [data/nocharacters](data/nocharacters/), and [data/regular](data/regular/) are JSON-preprocessed versions of the __MovieCoref__ corpus, which are more convenient for programmatic usage.

- The _addsays_ version adds the word "says" between speakers and their utterance.
- The _nocharacters_ version omits the speakers entirely from the screenplay.
- The _regular_ version does not make any changes.

We recommend using the _regular_ version.

## Training

### Set up

Create a python 3 conda environment from the `env.yml` file.

```
conda env create --file env.yml
```

### Inter-rater Agreement

Find the interrater agreement of the annotators that labeled the __MovieCoref__ corpus.
The annotations on the validation set used to calculate the interrater agreement scores can be found in the [data/validation](data/validation/) directory.

```
python rater.py
```

### Training

Preprocess the screenplay documents and the annotation labels.
This creates preprocessed JSON files.

```
python preprocess.py --gold
```

## Prediction

Given a movie script for which you want to find the character coreference clusters, you need to first parse it to
get the line-level tags.
You can use [https://github.com/usc-sail/mica-screenplay-parser]()

``

``

## Citation

The formatted citation is:
```
Sabyasachee Baruah and Shrikanth Narayanan. 2023. Character Coreference Resolution in Movie Screenplays.
In Findings of the Association for Computational Linguistics: ACL 2023, pages 10300–10313, Toronto, Canada.
Association for Computational Linguistics.
```

The bibtex is:
```bibtex
@inproceedings{baruah-narayanan-2023-character,
    title = "Character Coreference Resolution in Movie Screenplays",
    author = "Baruah, Sabyasachee  and
      Narayanan, Shrikanth",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.654",
    doi = "10.18653/v1/2023.findings-acl.654",
    pages = "10300--10313",
}
```