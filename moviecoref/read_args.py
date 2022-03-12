# author : Sabyasachee
# argument parser

'''
Movie Script Coreference Resolution

Usage:
    moviecoref annotator
'''

# third party
from docopt import docopt

def read_args():
    cmd_args = docopt(__doc__)
    args = {}
    args["mode"] = "annotator"
    return args