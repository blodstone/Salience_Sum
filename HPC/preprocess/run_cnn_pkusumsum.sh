#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o run_cnn_pkusumsum.txt
#$ -wd /home/acp16hh/Salience_Sum

module load apps/python/conda
source activate gwen

python HPC/parallel_split_cnn_pkusumsum.py -input_path /data/acp16hh/data/cnn/raw/docs -tmp_path /data/acp16hh/data/cnn/intermediary/raw -pkusumsum_path /home/acp16hh/PKUSUMSUM -n 75000
