#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
HOME=/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum
DATA=$HOME/data/bbc
source $HOME/venv/bin/activate

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_16/200
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/batch_16/seq2seq_salience_emb_mlp.jsonnet -output_path $DATA/result_3/seq2seq_salience_emb_mlp_200.out -batch_size 24 --cuda

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_16/300
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/batch_16/seq2seq_salience_emb_mlp.jsonnet -output_path $DATA/result_3/seq2seq_salience_emb_mlp_300.out -batch_size 24 --cuda

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/100
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/batch_16/seq2seq_salience_emb_mlp_attn.jsonnet -output_path $DATA/result_3/seq2seq_salience_emb_mlp_attn_100.out -batch_size 24 --cuda

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/200
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/batch_16/seq2seq_salience_emb_mlp_attn.jsonnet -output_path $DATA/result_3/seq2seq_salience_emb_mlp_attn_200.out -batch_size 24 --cuda

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/300
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -module pg_salience_feature -input $DATA/ready/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $HOME/HPC/pg_salience_feature/batch_16/seq2seq_salience_emb_mlp_attn.jsonnet -output_path $DATA/result_3/seq2seq_salience_emb_mlp_attn_200.out -batch_size 24 --cuda
