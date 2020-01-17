#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=16G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -o /data/acp16hh/Exp_Gwen_Saliency/crf/crf_summ/o.txt
#$ -e /data/acp16hh/Exp_Gwen_Saliency/crf/crf_summ/e.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout master
MODEL=/data/acp16hh/Exp_Gwen_Saliency/crf/crf_summ
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

allennlp train -s $MODEL -f --include_package salience_sum_crf HPC/crf/crf_summ.jsonnet
