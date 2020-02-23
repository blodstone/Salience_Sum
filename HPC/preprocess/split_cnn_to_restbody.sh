#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o split_cnn_to_restbody.txt
#$ -wd /home/acp16hh/Salience_Sum

module load apps/python/conda
source activate gwen

python preprocess/convert/split_cnn_to_restbody.py -input_path /data/acp16hh/data/cnn/ready -output_path /data/acp16hh/data/cnn/raw
