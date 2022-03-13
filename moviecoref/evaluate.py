# author : Sabyaschee

# standard library
import os
import sys
from typing import List

# third party
from moviecoref.parse_scripts_noindent import parse

def evaluate(movies: List[str], speaker_word="says", keep_speakers=True, keep_persons=False, merge_speakers=True, pronoun_resolution=False):
    
    annotated_movies = [line.split()[0] for line in open("data/annotation/movies.txt").read().splitlines()]
    parse_dir = "data/annotation/parsed"

    for movie in movies:

        #####################################################################
        #### check if movie is annotated and if its script exists
        #####################################################################
        
        if not movie in annotated_movies:
            print("{} not in list of annotated movies".format(movie))
            sys.exit(1)
        
        script_file = os.path.join("data/annotation/screenplay/{}.txt".format(movie))
        if not os.path.exists(script_file):
            print("{} script not found".format(movie))
            sys.exit(1)

        #####################################################################
        #### create parsed screenplay directory if it does not exist
        #####################################################################
        
        os.makedirs(parse_dir, exist_ok=True)

        #####################################################################
        #### parse movie screenplay
        #####################################################################
        
        parse(script_file, parse_dir, "off", "on", "off")