#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=32G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -P rse
#$ -q rse.q
#$ -o pg_salience_feature_sum_16_dgx.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout dev
MODEL=/data/acp16hh/Exp_Gwen_Saliency/pg_salience_feature/sum_16_dgx
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

python summarize -input /data/acp16hh/data/bbc/ready/all/train.sum.tsv -vocab_path $MODEL/vocabulary -model $MODEL/best.th -model_config HPC/dgx/pg_salience_feature/pg_salience_feature_sum_16.jsonnet -output_path $MODEL -module pg_salience_feature

cd postprocess

python rouge.py -r /data/acp16hh/data/bbc/ready/all/test.sum.tsv -c $MODEL/test.sum.out
