#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=24G
#$ -l h_rt=48:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o oracle_bert_cnn.txt
#$ -wd /home/acp16hh/Salience_Sum
module load apps/python/conda
source activate gwen

python bert_extractive/oracle.py -input /data/acp16hh/data/cnn/ready/test.tsv -output /data/acp16hh/data/cnn/result -dataset CNN
