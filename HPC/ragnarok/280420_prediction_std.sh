#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
HOME=/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum
DATA=$HOME/data/bbc

source $HOME/venv/bin/activate

MODEL=$HOME/model/bbc/pg_clean_16/100
CONFIG=$HOME/model_config/pg_clean_16/std_beam/100
python $HOME/summarize.py -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $CONFIG/config.json -output_path $DATA/result_4/std_pg_clean_100.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/pg_clean_16/200
CONFIG=$HOME/model_config/pg_clean_16/std_beam/200
python $HOME/summarize.py -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $CONFIG/config.json -output_path $DATA/result_4/std_pg_clean_200.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/pg_clean_16/300
CONFIG=$HOME/model_config/pg_clean_16/std_beam/300
python $HOME/summarize.py -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $CONFIG/config.json -output_path $DATA/result_4/std_pg_clean_300.out -batch_size 24 --cuda --use_salience

