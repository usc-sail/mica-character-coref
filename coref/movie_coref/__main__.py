# author : Sabyasachee

# custom
from moviecoref.read_args import read_args
from moviecoref.agreement import interrater_agreement
from moviecoref.evaluate import evaluate

args = read_args()

if args["mode"] == "agreement":
    interrater_agreement()

elif args["mode"] == "evaluate":
    if args["movie"] == "all":
        movies = [line.split()[0] for line in open("data/annotation/movies.txt").read().splitlines()]
    else:
        movies = [args["movie"]]
    evaluate(movies, args["spk"], args["speaker"], args["person"], args["merge"], args["pronoun"])