#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=32G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -P rse
#$ -q rse.q
#$ -o pg_salience_bilinear_16.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout dev
MODEL=/data/acp16hh/Exp_Gwen_Saliency/pg_salience_feature/bilinear_16
module load apps/python/conda
module load libs/cudnn/7.6.5.32/binary-cuda-10.0.130
source activate gwen

allennlp train -s $MODEL -f --file-friendly-logging --include-package pg_salience_feature HPC/dgx/pg_salience_feature/pg_salience_bilinear_16.jsonnet
