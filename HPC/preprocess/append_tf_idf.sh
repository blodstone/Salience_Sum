#!/usr/bin/env bash
#$ -l rmem=120G
#$ -l h_rt=128:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o append_tf_idf.txt
#$ -wd /home/acp16hh/Salience_Sum
module load apps/python/conda
source activate gwen

HOME=/home/acp16hh/Salience_Sum

python preprocess/preprocess_salience_from_pkupu.py -set train test validation - /data/acp16hh/data/bbc/raw/bbc-tokenized-segmented-final/restbody -golds_pku /data/acp16hh/data/bbc/raw/bbc-tokenized-segmented-final/firstsentence -output /data/acp16hh/data/bbc/ready --tfidf -extra_name tfidf -index /data/acp16hh/data/bbc/raw/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json
