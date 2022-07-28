#!/bin/bash

python map_annotation_to_parsed_script.py -a data/annotation/bourne.coref.csv -s data/annotation/bourne.script.txt -p data/annotation/bourne.script_parsed.txt -o data/annotation/bourne.coref.mapped.csv &
python map_annotation_to_parsed_script.py -a data/annotation/basterds.coref.csv -s data/annotation/basterds.script.txt -p data/annotation/basterds.script_parsed.txt -o data/annotation/basterds.coref.mapped.csv &
python map_annotation_to_parsed_script.py -a data/annotation/shawshank.coref.csv -s data/annotation/shawshank.script.txt -p data/annotation/shawshank.script_parsed.txt -o data/annotation/shawshank.coref.mapped.csv