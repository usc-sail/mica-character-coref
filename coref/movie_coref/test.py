import json
import jsonlines
# json.load(open("/home/sbaruah_usc_edu/mica_text_coref/data/temp/movie_dict.json"))
with jsonlines.open("/home/sbaruah_usc_edu/mica_text_coref/data/movie_coref/results/regular/movie.jsonlines") as reader:
    for obj in reader:
        print(obj["movie"], obj["rater"])
        print(len(obj["token"]))
        print(obj["clusters"].keys())