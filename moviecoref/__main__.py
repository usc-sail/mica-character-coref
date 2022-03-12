# author : Sabyasachee

# custom
from moviecoref.read_args import read_args
from moviecoref.annotator_evaluation import interrater_agreement

args = read_args()
if args["mode"] == "annotator":
    interrater_agreement()
else:
    print("wrong mode")