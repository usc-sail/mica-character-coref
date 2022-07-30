#!/bin/bash

# any error will cause the script to exit immediately
set -e

# check arguments
if [ $# -lt 3 ]; then
	echo -e "Usage:\n./prepare_data.sh <data_directory> <ontonotes_directory> <scripts_directory>"
	exit
fi

# check if python2 is used
if python -c "import sys; sys.exit(int(sys.version.startswith('2.7')))"; then
	echo "CONLL-2012 scripts require python 2.7!"
	echo "Activate python=2.7 environment and run the script again"
	exit
fi

# variables
data_directory=$1
ontonotes_directory=$2
scripts_directory=$3
conll_url=http://conll.cemantix.org/2012/download

echo "data directory =" $data_directory
echo "ontonotes directory =" $ontonotes_directory
echo "scripts directory =" $scripts_directory
echo

# function to download archive, uncompress it, and remove archive file
dlx() {
	wget $1/$2
	tar -xvzf $2 -C $data_directory
	rm $2
}

# download conll-2012 train, development, and test skel files, and perl coreference scorer
download_conll_2012() {
	dlx $conll_url conll-2012-train.v4.tar.gz
	dlx $conll_url conll-2012-development.v4.tar.gz
	dlx $conll_url/test conll-2012-test-key.tar.gz
	dlx $conll_url/test conll-2012-test-official.v9.tar.gz
	dlx $conll_url conll-2012-scripts.v3.tar.gz
	dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz

	# moving perl coreference scorer to scripts
	mv $data_directory/reference-coreference-scorers/v8.01 $3/scorer
	rm -rf $data_directory/reference-coreference-scorers
}

# convert conll-2012 skeleton files to conll files
convert_skeleton_to_conll() {
	bash $data_directory/conll-2012/v3/scripts/skeleton2conll.sh \
			 -D $ontonotes_directory/data/files/data \
			 $data_directory/conll-2012
}

# compile language/partition
compile_partition() {
  rm -f $2.$5.$3$4
  cat $data_directory/conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> \
			$data_directory/conll-2012/gold/$2.$5.$3$4
}

# compile language
compile_language() {
  compile_partition development dev v4 _gold_conll $1
  compile_partition train train v4 _gold_conll $1
  compile_partition test test v4 _gold_conll $1
}

# compile all languages
compile_languages() {
	mkdir -p $data_directory/conll-2012/gold
	compile_language english
	compile_language chinese
	compile_language arabic
}

# download_conll_2012
# convert_skeleton_to_conll
# compile_languages
# python minimize.py
# python get_char_vocab.py

# python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines
# python cache_elmo.py train.english.jsonlines dev.english.jsonlines