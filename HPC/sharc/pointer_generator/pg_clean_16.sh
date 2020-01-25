#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=32G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o pg_clean_16.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout dev
export CUDA_VISIBLE_DEVICES=0
MODEL=/data/acp16hh/Exp_Gwen_Saliency/pg_clean_16
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

allennlp train -s $MODEL -f --file-friendly-logging --include-package pointer_generator_salience HPC/sharc/pointer_generator/pg_clean_16.jsonnet
