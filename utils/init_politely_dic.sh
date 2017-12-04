#!/usr/bin/bash

source ../../myvenv/bin/activate

DICTIONARIES=("sent_nouns.dic" "sent_verb_adj.dic")
for dic in ${DICTIONARIES[@]}; do
  echo "saving ${dic}"
  time python init_collections_from_file.py -d"\t" -f ${dic} --HEADER Y -m labelToScore
done

echo "finished saving dictionaries"
