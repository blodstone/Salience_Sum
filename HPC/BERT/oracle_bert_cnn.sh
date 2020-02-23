#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=24G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o oracle_bert_cnn.txt
#$ -wd /home/acp16hh/Salience_Sum
module load apps/python/conda
module load libs/cudnn/7.6.5.32/binary-cuda-10.0.130
source activate gwen

python bert_extractive/oracle.py -input /data/acp16hh/data/cnn/ready/test.tsv -output /data/acp16hh/data/cnn/result -dataset CNN
