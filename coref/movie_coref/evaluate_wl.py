# author : Sabyaschee

# standard library
import collections
import os

# third party
import jsonlines

# user
import scores

def evaluate_wl(results_folder, says=False):

    #####################################################################
    #### read input and output jsonlines
    #####################################################################
    
    if says:
        input_file = os.path.join(results_folder, "input/with_says/movie_with_says_coref.jsonlines")
        output_file = os.path.join(results_folder, "output/with_says/movie_with_says_coref.wl.output.jsonlines")
    else:
        input_file = os.path.join(results_folder, "input/normal/movie_coref.jsonlines")
        output_file = os.path.join(results_folder, "output/normal/movie_coref.wl.output.jsonlines")

    with jsonlines.open(input_file) as reader:
        coref_input = {}
        for data in reader:
            coref_input[data["movie"]] = data

    with jsonlines.open(output_file) as reader:
        coref_output = {}
        for data in reader:
            movie = data["document_id"][3:]
            coref_output[movie] = data
    
    #####################################################################
    #### get movies list
    #####################################################################

    movies = sorted(set(coref_input.keys()).intersection(coref_output.keys()))

    all_gold_clusters = []
    all_pred_clusters = []

    for movie in movies:
        movie_data = coref_input[movie]
        movie_wl = coref_output[movie]

        movie = movie_data["movie"]
        tags = movie_data["parse"]
        tokens = movie_data["token"]
        begins, ends, characters = movie_data["begin"], movie_data["end"], movie_data["character"]

        n_gold_clusters = len(set(movie_data["character"]))
        n_pred_clusters = len(movie_wl["span_clusters"])

        print(movie)
        print(f"\t{n_gold_clusters} gold clusters")
        print(f"\t{n_pred_clusters} pred clusters")

        #####################################################################
        #### filter non-character clusters
        #####################################################################

        pred_clusters = []

        for cluster in movie_wl["span_clusters"]:
            for begin, end in cluster:
                if all(tag == "C" for tag in tags[begin: end]):
                    pred_clusters.append(cluster)
                    break
        
        print(f"\t{len(pred_clusters)} pred character clusters")

        #####################################################################
        #### merge speakers
        #####################################################################

        to_merge_cluster_indexes = set((i,) for i in range(len(pred_clusters)))

        for i in range(len(pred_clusters)):
            for j in range(i + 1, len(pred_clusters)):

                merge = False

                for a, b in pred_clusters[i]:
                    if all(tag == "C" for tag in tags[a: b]):

                        for x, y in pred_clusters[j]:
                            if all(tag == "C" for tag in tags[x: y]):

                                if tokens[a: b] == tokens[x: y]:
                                    merge = True
                                    break

                        if merge:
                            break
                
                if merge:
                    cluster_i = next(cluster_index for cluster_index in to_merge_cluster_indexes if i in cluster_index)
                    cluster_j = next(cluster_index for cluster_index in to_merge_cluster_indexes if j in cluster_index)
                    
                    if cluster_i != cluster_j:
                        cluster_ij = [ci for ci in cluster_i] + [cj for cj in cluster_j]
                        cluster_ij = tuple(sorted(cluster_ij))

                        to_merge_cluster_indexes.discard(cluster_i)
                        to_merge_cluster_indexes.discard(cluster_j)
                        to_merge_cluster_indexes.add(cluster_ij)
        
        merged_pred_clusters = []

        for cluster_index in to_merge_cluster_indexes:
            merged_wl_cluster = []
            for i in cluster_index:
                merged_wl_cluster += pred_clusters[i]
            merged_pred_clusters.append(merged_wl_cluster)
        
        print(f"\t{len(merged_pred_clusters)} pred speaker-merged character clusters")

        pred_clusters = merged_pred_clusters

        #####################################################################
        #### get gold clusters
        #####################################################################

        gold_clusters = collections.defaultdict(set)

        for begin, end, character in zip(begins, ends, characters):
            gold_clusters[character].add((begin, end + 1))
        
        gold_clusters = list(gold_clusters.values())

        #####################################################################
        #### get pred clusters
        #####################################################################

        pred_clusters = [set([(begin, end) for begin, end in cluster]) for cluster in pred_clusters]

        #####################################################################
        #### score
        #####################################################################

        conllF1 = scores.conll2012(gold_clusters, pred_clusters)
        print(f"\tconll F1 = {conllF1:.3f}")

        #####################################################################
        #### collect gold and pred clusters
        #####################################################################
        
        for cluster in gold_clusters:
            new_cluster = set((movie, begin, end) for begin, end in cluster)
            all_gold_clusters.append(new_cluster)
        
        for cluster in pred_clusters:
            new_cluster = set((movie, begin, end) for begin, end in cluster)
            all_pred_clusters.append(new_cluster)

        print()

    all_conllF1 = scores.conll2012(all_gold_clusters, all_pred_clusters)
    print(f"micro conllF1 = {all_conllF1:.3f}")

if __name__=="__main__":
    evaluate_wl("/workspace/mica-text-coref/results")
    print("\n\n")
    evaluate_wl("/workspace/mica-text-coref/results", says=True)