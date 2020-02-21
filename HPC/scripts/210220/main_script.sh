#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
module load apps/python/conda
source activate gwen

python ../../automate_hpc.py --all -spec_file run_specs.csv
