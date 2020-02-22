#!/bin/bash
export RANDOM_SEED=100
export NUMPY_SEED=100
export PYTORCH_SEED=100
export CUDA_VISIBLE_DEVICES=0
HOME=/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum
MODEL=$HOME/data/output/pg_salience_feature/bbc/dgx/salience_emb_attn_16/100
DATA=$HOME/data/bbc
source $HOME/venv/bin/activate
export train_path=$DATA/ready/train.salience.tsv
export validation_path=$DATA/ready/validation.salience.tsv
allennlp train -s $MODEL -f --file-friendly-logging --include-package pg_salience_feature $HOME/HPC/pg_salience_feature/pg_salience_emb_mlp_attn_16.jsonnet
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/pg_salience_emb_mlp_attn_16.jsonnet -output_path $DATA/result/test_pg_salience_feature_bbc_dgx_salience_emb_attn_16.out -batch_size 24
