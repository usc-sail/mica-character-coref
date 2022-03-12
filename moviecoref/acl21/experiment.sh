#!/bin/bash

for SIM in 0.5 0.6 0.7 0.8 0.9 1
do
    for MERGE in 0 1 2 3 4 5
    do
        python evaluate_all_by_joining_elements.py --results results/coreference_evaluation-SIM$SIM-MERGE$MERGE.csv --min_speaker_similarity $SIM --max_speaker_merges $MERGE
    done
done