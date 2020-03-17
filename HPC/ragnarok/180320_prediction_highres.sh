#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
HOME=/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum
DATA=$HOME/data/bbc
source $HOME/venv/bin/activate

python $HOME/preprocess/convert/tag_highres.py

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_16/200
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -input $DATA/ready/highres/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_highres/seq2seq_salience_emb_mlp_200.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_16/300
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -input $DATA/ready/highres/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_highres/seq2seq_salience_emb_mlp_300.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/200
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -input $DATA/ready/highres/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_highres/seq2seq_salience_emb_mlp_attn_200.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/seq2seq_emb_mlp_attn_16/300
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -input $DATA/ready/highres/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_highres/seq2seq_salience_emb_mlp_attn_300.out -batch_size 24 --cuda --use_salience

MODEL=$HOME/model/bbc/seq2seq_clean_16/300
python $HOME/postprocess/retrieve_last_model.py $MODEL
python $HOME/summarize.py -input $DATA/ready/highres/test.salience.tsv -vocab_path $MODEL/vocabulary -model $MODEL/pick.th -model_config $MODEL/config.json -output_path $DATA/result_highres/seq2seq_clean_16_300.out -batch_size 24 --cuda

python postprocess/rouge.py -f ../data/bbc/result_highres -r ../data/bbc/ready/test.salience.tsv -o ../data/output/ -n result_seq2seq_highres.out
