#!/bin/bash

# screenplays=(shawshank basterds bourne)

# for screenplay in ${screenplays[@]}
# do
#     echo $screenplay
#     python screenplay_parser.py -i data/coreference/$screenplay.script.txt -o data/coreference/
#     python map_annotation_to_parsed_script.py -a data/coreference/$screenplay.coref.csv -s data/coreference/$screenplay.script.txt -p data/coreference/$screenplay.script_parsed.txt -o data/coreference/$screenplay.coref.mapped.csv
#     echo
# done

python evaluate.py
python print_evaluation_numbers.py