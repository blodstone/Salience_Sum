#!/usr/binew_datan/env bash

# Collect all the data and store it into a pickle for later processing
cd /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum
source venv/bin/activate
python postprocess/prediction_process/collect_data.py -input_folder /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/prediction_output -output_folder /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/postprocess_data -src /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc/ready/test.salience.tsv
# First we check which instances in both the salience and non-salience prediction that are differs in ROUGE score.
export INPUT_PATH='/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/postprocess_data'
# Input: folder of out
# Output: csv

