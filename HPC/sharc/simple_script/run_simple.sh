#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
#$ -wd /home/acp16hh/Salience_Sum/HPC/sharc/simple_script
module load apps/python/conda
source activate gwen
python run_simple.py