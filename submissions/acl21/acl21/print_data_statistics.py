import pandas as pd
import spacy
import re

def print_data_statistics(coref_file, script_file):
    coref_df = pd.read_csv(coref_file, index_col=None)
    script = open(script_file).read().strip()
    script = re.sub("\s+", " ", script)
    spacy_nlp = spacy.load("en_core_web_sm")

    script_spacy_doc = spacy_nlp(script)
    n_tokens = len(script_spacy_doc)

    n_characters = coref_df.entityLabel.unique().size
    n_speaking_characters = coref_df[coref_df.SPEAKER].entityLabel.unique().size
    n_multiton_characters = (coref_df.groupby("entityLabel").agg(len).entityNum > 1).sum()
    n_multiton_speaking_characters = (coref_df[coref_df.SPEAKER].groupby("entityLabel").agg(len).entityNum > 1).sum()

    pronouns = "I, me, my, mine, myself, We, us, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themself, themselves".lower().split(", ")
    coref_df.PRONOUN |= coref_df.surface.str.lower().isin(pronouns)

    n_named_mentions = (~coref_df.PRONOUN & ~coref_df.NOMINAL).sum()
    n_pronominal_mentions = (coref_df.PRONOUN).sum()
    n_nominal_mentions = (coref_df.NOMINAL).sum()
    n_difficult_mentions = (coref_df.DIFFICULT).sum()

    print(f"#tokens = {n_tokens}")
    print("Characters")
    print(f"\t#characters = {n_characters}")
    print(f"\t#speaking-characters = {n_speaking_characters}")
    print(f"\t#multiton-characters = {n_multiton_characters}")
    print(f"\t#multiton-speaking-characters = {n_multiton_speaking_characters}")
    print("Mentions")
    print(f"\t#named-mentions of characters = {n_named_mentions}")
    print(f"\t#pronominal-mentions of characters = {n_pronominal_mentions}")
    print(f"\t#nominal-mentions of characters = {n_nominal_mentions}")
    print(f"\t#difficult-mentions of characters = {n_difficult_mentions}")

def print_data_statistics_all():
    print("The Shawshank Redemption:")
    print_data_statistics("data/coreference/shawshank.coref.csv", "data/coreference/shawshank.script.txt")
    print()

    print("Bourne Ultimatum:")
    print_data_statistics("data/coreference/bourne.coref.csv", "data/coreference/bourne.script.txt")
    print()

    print("Inglourious Basterds:")
    print_data_statistics("data/coreference/basterds.coref.csv", "data/coreference/basterds.script.txt")

if __name__ == "__main__":
    print_data_statistics_all()