#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=16G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -wd /home/acp16hh/Salience_Sum
#$ -e /data/acp16hh/Exp_Gwen_Saliency/crf.e
#$ -o /data/acp16hh/Exp_Gwen_Saliency/crf.o
git checkout master
MODEL=/fastdata/acp16hh/Exp_Gwen_Saliency

mkdir -p $DATA
mkdir -p $LOG
mkdir -p $MODEL
mkdir -p $OUTPUT

module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

allennlp train -s $MODEL/crf --include-package salience_sum_crf -r