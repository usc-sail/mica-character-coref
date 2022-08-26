import os
import json
import io
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_directory", None,
  "Path to the conll-2012 gold directory",
 required=True)

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for sentence in json.loads(line)["sentences"]:
          for word in sentence:
            vocab.update(word)
  vocab = sorted(list(vocab))
  with io.open(output_filename, mode="w", encoding="utf8") as f:
    for char in vocab:
      f.write(char)
      f.write(u"\n")
  print("Wrote {} characters to {}".format(len(vocab), output_filename))

def get_char_vocab_language(language):
  conll_directory = FLAGS.conll_directory
  get_char_vocab([os.path.join(conll_directory, 
  "{}.{}.jsonlines".format(partition, language))
   for partition in ("train", "dev", "test")], os.path.join(
    conll_directory, "char_vocab.{}.txt".format(language)))

def get_char_vocab_languages():
  get_char_vocab_language("english")
  get_char_vocab_language("chinese")
  get_char_vocab_language("arabic")

def main(argv):
  get_char_vocab_languages()

if __name__=="__main__":
  app.run(main)