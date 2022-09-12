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

1. Create python 3 conda environment. <br>
  `conda create --name coreference python=3.10`.

2. Activate the conda environment. <br> 
  `conda activate coreference`

3. Install libraries. <br>
  `conda install -c conda-forge jsonlines absl-py unidecode ipywidgets ipykernel gpustat`

4. Install pytorch. <br> 
  `conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`

5. Install huggingface transformers library. <br> 
  `conda install -c huggingface transformers`

6. Install the python scorch library, which is used for coreference evaluation. <br> 
  `pip install scorch`

7. Create python 2 conda environment, which is required for the conll-2012 scripts. <br>
  `conda create --name py2 python=2`

## Setting up the data

1. Create data and tensors directory. <br>
  `mkdir data`
  `mkdir data/tensors/longformer_seq_tensors`

2. Download the ontonotes v5 corpus and save it to the data directory at `data/ontonotes-release-5.0`.

3. Activate the python 2 conda environment. <br>
  `conda activate py2`

4. Download the conll-2012 data and scripts, and create the gold conll files. <br>
  `sh coref/seq_coref/prepare_data.sh data data/ontonotes-release-5.0 coref/seq_coref`

5. Activate the python 3 conda environment. <br>
  `conda activate coreference`

6. Convert the gold conll files to jsonlines. <br>
  `python coref/seq_coref/minimize.py --conll_directory=data/conll-2012/gold`

7. Create tensors from the jsonlines files for training and testing. <br>
  `python coref/seq_coref/tensorize_main.py --conll_directory=data/conll-2012/gold --longformer_tensors_directory=data/tensors/longformer_seq_tensors`