# author : Sabyasachee
# argument parser

'''
Movie Script Coreference Resolution

Usage:
    moviecoref agreement
    moviecoref evaluate [--movie=<movie> --spk=<word> --speaker --person --merge --pronoun]

Options:
    -h, --help                          Show this screen and exit

        --movie=<movie>                 evaluate coreference in given annotated movie [default: all]
        --spk=<word>                    insert speaker word between speaker and utternace [default: ]
        --speaker                       retain speakers
        --person                        retain persons found using spacy NER
        --merge                         merge clusters of same speaker
        --pronoun                       use heuristic pronoun resolution
'''

# third party
from docopt import docopt

def read_args():
    cmd_args = docopt(__doc__)
    args = {}

    if cmd_args["agreement"]:
        args["mode"] = "agreement"

    else:
        args["mode"] = "evaluate"
        args["movie"] = cmd_args["--movie"]
        args["spk"] = cmd_args["--spk"]
        args["speaker"] = cmd_args["--speaker"]
        args["person"] = cmd_args["--person"]
        args["merge"] = cmd_args["--merge"]
        args["pronoun"] = cmd_args["--pronoun"]
    
    return args