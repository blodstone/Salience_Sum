#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -o run_cnn_pkusumsum_build_tsv.txt
#$ -wd /home/acp16hh/Salience_Sum

module load apps/python/conda
source activate gwen
DRMAA_LIBRARY_PATH=/usr/local/sge/live/lib/lx-amd64/libdrmaa.so; export DRMAA_LIBRARY_PATH
cd preprocess
RAW=/data/acp16hh/data/cnn/raw
python preprocess_salience_from_pkupu.py -set train validation test -index $RAW/CNN-TRAINING-DEV-TEST-SPLIT.json
