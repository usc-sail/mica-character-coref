# Coreference in Movie Scripts

This repository contains the source code and data for the paper
[Annotation and Evaluation of Coreference Resolution in Screenplays](https://aclanthology.org/2021.findings-acl.176/)
in ACL Findings 2021

Currently, we are developing a coreference model for movie screenplay documents.

## Citation
Please cite the following paper if you found it useful. Thanks:)

```
@inproceedings{baruah-etal-2021-annotation,
    title = "Annotation and Evaluation of Coreference Resolution in Screenplays",
    author = "Baruah, Sabyasachee  and
      Nallan Chakravarthula, Sandeep  and
      Narayanan, Shrikanth",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.176",
    doi = "10.18653/v1/2021.findings-acl.176",
    pages = "2004--2010",
}
```

# Sequence-Based Coreference Resolution

## Setting up the conda environment

Create python 3 conda environment from the `env.yml` file, and a python 2 conda
environment which is required for the conll-2012 scripts.
```
conda env create --file env.yml
conda create --name py2 python=2
```

## Setting up the data

1. Create data and tensors directory. <br>
    ```
    mkdir data
    mkdir data/tensors/longformer_seq_tensors
    ```

2. Download the ontonotes v5 corpus and save it to the data directory at 
  `data/ontonotes-release-5.0`.

3. Activate the python 2 conda environment, and run 
  `coref/seq_coref/setup.sh` script which downloads the conll-2012 data
    and creates gold conll files. <br>
    ```
    conda activate py2
    sh coref/seq_coref/setup.sh data data/ontonotes-release-5.0 coref/seq_coref
    ```

4. Activate the python 3 conda environment, convert the gold conll files to
  jsonlines, and create tensors from the jsonlines files.<br>
    ```
    conda activate coreference
    python coref/seq_coref/minimize.py --conll_directory=data/conll-2012/gold
    python coref/seq_coref/tensorize_main.py --conll_directory=data/conll-2012/gold --longformer_tensors_directory=data/tensors/longformer_seq_tensors
    ```

# Movie Coreference Resolution

We are going to fine-tune the 
[Word-Level Coreference Model](https://aclanthology.org/2021.emnlp-main.605/)
using their their repository
[vdobrovolskii/wl-coref](https://github.com/vdobrovolskii/wl-coref).

## Citation

We use the following work

```
@inproceedings{dobrovolskii-2021-word,
    title = "Word-Level Coreference Resolution",
    author = "Dobrovolskii, Vladimir",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.605",
    doi = "10.18653/v1/2021.emnlp-main.605",
    pages = "7670--7675",
}
```