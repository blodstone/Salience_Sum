#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=32G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -P rse
#$ -q rse.q
#$ -o pg_salience_emb_mlp_16.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout dev
MODEL=/data/acp16hh/Exp_Gwen_Saliency/pg_salience_feature/emb_mlp_16_2
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

allennlp train -s $MODEL -f --file-friendly-logging --include-package pg_salience_feature HPC/dgx/pg_salience_feature/pg_salience_emb_mlp_16.jsonnet
