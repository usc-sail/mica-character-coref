#!/bin/bash

# echo "avengers_endgame"
# python predict.py roberta ../../results/input/normal/avengers_endgame_coref.wl.jsonlines ../../results/output/normal/avengers_endgame_coref.wl.output.jsonlines

# echo "dead_poets_society"
# python predict.py roberta ../../results/input/normal/dead_poets_society_coref.wl.jsonlines ../../results/output/normal/dead_poets_society_coref.wl.output.jsonlines

# echo "john_wick"
# python predict.py roberta ../../results/input/normal/john_wick_coref.wl.jsonlines ../../results/output/normal/john_wick_coref.wl.output.jsonlines

# echo "prestige"
# python predict.py roberta ../../results/input/normal/prestige_coref.wl.jsonlines ../../results/output/normal/prestige_coref.wl.output.jsonlines

# echo "quiet_place"
# python predict.py roberta ../../results/input/normal/quiet_place_coref.wl.jsonlines ../../results/output/normal/quiet_place_coref.wl.output.jsonlines

# echo "zootopia"
# python predict.py roberta ../../results/input/normal/zootopia_coref.wl.jsonlines ../../results/output/normal/zootopia_coref.wl.output.jsonlines

# echo "shawshank"
# python predict.py roberta ../../results/input/normal/shawshank_coref.wl.jsonlines ../../results/output/normal/shawshank_coref.wl.output.jsonlines

# echo "bourne"
# python predict.py roberta ../../results/input/normal/bourne_coref.wl.jsonlines ../../results/output/normal/bourne_coref.wl.output.jsonlines

# echo "basterds"
# python predict.py roberta ../../results/input/normal/basterds_coref.wl.jsonlines ../../results/output/normal/basterds_coref.wl.output.jsonlines









echo "avengers_endgame"
python predict.py roberta ../../results/input/with_says/avengers_endgame_with_says_coref.wl.jsonlines ../../results/output/with_says/avengers_endgame_with_says_coref.wl.output.jsonlines

echo "dead_poets_society"
python predict.py roberta ../../results/input/with_says/dead_poets_society_with_says_coref.wl.jsonlines ../../results/output/with_says/dead_poets_society_with_says_coref.wl.output.jsonlines

echo "john_wick"
python predict.py roberta ../../results/input/with_says/john_wick_with_says_coref.wl.jsonlines ../../results/output/with_says/john_wick_with_says_coref.wl.output.jsonlines

echo "prestige"
python predict.py roberta ../../results/input/with_says/prestige_with_says_coref.wl.jsonlines ../../results/output/with_says/prestige_with_says_coref.wl.output.jsonlines

echo "quiet_place"
python predict.py roberta ../../results/input/with_says/quiet_place_with_says_coref.wl.jsonlines ../../results/output/with_says/quiet_place_with_says_coref.wl.output.jsonlines

echo "zootopia"
python predict.py roberta ../../results/input/with_says/zootopia_with_says_coref.wl.jsonlines ../../results/output/with_says/zootopia_with_says_coref.wl.output.jsonlines

echo "shawshank"
python predict.py roberta ../../results/input/with_says/shawshank_with_says_coref.wl.jsonlines ../../results/output/with_says/shawshank_with_says_coref.wl.output.jsonlines

echo "bourne"
python predict.py roberta ../../results/input/with_says/bourne_with_says_coref.wl.jsonlines ../../results/output/with_says/bourne_with_says_coref.wl.output.jsonlines

echo "basterds"
python predict.py roberta ../../results/input/with_says/basterds_with_says_coref.wl.jsonlines ../../results/output/with_says/basterds_with_says_coref.wl.output.jsonlines