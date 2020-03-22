#!/usr/bin/env bash
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -j y
module load apps/python/conda
source activate gwen
DRMAA_LIBRARY_PATH=/usr/local/sge/live/lib/lx-amd64/libdrmaa.so; export DRMAA_LIBRARY_PATH

python ../../automate_hpc.py -spec_file run_specs.csv -data_path /data/acp16hh/data/ --dgx --last_summary --force
