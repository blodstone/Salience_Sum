#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
HOME=/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum
DATA=$HOME/data/bbc
source $HOME/venv/bin/activate

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/300
python $HOME/summarize.py -input $DATA/ready/test.constraint.10.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_4/constraint_seq2seq_salience_emb_mlp_attn_300.out -batch_size 24 --cuda --use_salience --use_constraint

