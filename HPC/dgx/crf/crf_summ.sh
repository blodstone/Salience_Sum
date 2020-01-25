#!/usr/bin/env bash
#$ -l gpu=1
#$ -l rmem=32G
#$ -l h_rt=96:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -P rse
#$ -q rse.q
#$ -j y
#$ -o crf_summ_dgx_log.txt
#$ -wd /home/acp16hh/Salience_Sum
git checkout dev
export CUDA_VISIBLE_DEVICES=0
MODEL=/data/acp16hh/Exp_Gwen_Saliency/crf/crf_summ_dgx
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate gwen

allennlp train -s $MODEL -f --file-friendly-logging --include-package salience_sum_crf HPC/dgx/crf/crf_summ.jsonnet
