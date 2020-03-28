# %%
import argparse
import pickle
import os
from collections import Counter

import pandas as pd
from pathlib import Path

from postprocess.prediction_process.collect_data import Data

if 'INPUT_PATH' in os.environ:
    input_folder = Path(os.environ['INPUT_PATH'])
else:
    input_folder = Path(
        '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/postprocess_data')

if 'OUTPUT_PATH' in os.environ:
    output_folder = Path(os.environ['OUTPUT_PATH'])
else:
    output_folder = Path('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/postprocess_result')

# %%
def count_salience(salience):
    salience = [[float(val) for val in sal] for sal in salience]
    salience = [sum(values) for values in zip(*salience)]
    c = Counter(salience)
    return c[0], c[1]+c[2]+c[3]+c[4]+c[5]+c[6], c[1]

def calculate_overlap_score(ref, salience, cand):
    salience = [[float(val) for val in sal] for sal in salience]
    salience = [sum(values) for values in zip(*salience)]
    pos_sal = [(token, sal) for token, sal in zip(ref, salience) if sal > 1]
    token_to_sal = {}
    for token, sal in pos_sal:
        if token in token_to_sal:
            if token_to_sal[token] < sal:
                token_to_sal[token] = sal
        else:
            token_to_sal[token] = sal
    scores = 0
    used_tokens = []
    for token in cand:
        if token in token_to_sal and token not in used_tokens:
            scores += 1
            used_tokens.append(token)
    return scores
#%%
rouge_scores = {}
for file in input_folder.iterdir():
    print(f'Processing {file.stem}')
    name = file.stem
    data = pickle.load(file.open('rb'))
    rouge_scores[f'{name}_r1_precision'] = []
    rouge_scores[f'{name}_r1_recall'] = []
    rouge_scores[f'{name}_r1_fmeasure'] = []
    rouge_scores[f'{name}_r2_precision'] = []
    rouge_scores[f'{name}_r2_recall'] = []
    rouge_scores[f'{name}_r2_fmeasure'] = []
    rouge_scores[f'{name}_rL_precision'] = []
    rouge_scores[f'{name}_rL_recall'] = []
    rouge_scores[f'{name}_rL_fmeasure'] = []
    rouge_scores[f'{name}_length_cand'] = []
    rouge_scores[f'{name}_length_doc'] = []
    rouge_scores[f'{name}_candidate'] = []
    rouge_scores[f'{name}_reference'] = []
    rouge_scores[f'{name}_salience_overlap_score_cand'] = []
    rouge_scores[f'{name}_salience_overlap_score_ref'] = []
    rouge_scores[f'{name}_no_salience_count'] = []
    rouge_scores[f'{name}_salience_count'] = []
    rouge_scores[f'{name}_salience_count_1'] = []
    for datum in data:
        rouge_scores[f'{name}_r1_precision'].append(datum.scores['rouge1']['precision'])
        rouge_scores[f'{name}_r1_recall'].append(datum.scores['rouge1']['recall'])
        rouge_scores[f'{name}_r1_fmeasure'].append(datum.scores['rouge1']['fmeasure'])
        rouge_scores[f'{name}_r2_precision'].append(datum.scores['rouge2']['precision'])
        rouge_scores[f'{name}_r2_recall'].append(datum.scores['rouge2']['recall'])
        rouge_scores[f'{name}_r2_fmeasure'].append(datum.scores['rouge2']['fmeasure'])
        rouge_scores[f'{name}_rL_precision'].append(datum.scores['rougeLsum']['precision'])
        rouge_scores[f'{name}_rL_recall'].append(datum.scores['rougeLsum']['recall'])
        rouge_scores[f'{name}_rL_fmeasure'].append(datum.scores['rougeLsum']['fmeasure'])
        rouge_scores[f'{name}_length_cand'].append(len(datum.cand))
        rouge_scores[f'{name}_length_doc'].append(len(datum.doc))
        rouge_scores[f'{name}_candidate'].append(' '.join(datum.cand))
        rouge_scores[f'{name}_reference'].append(' '.join(datum.ref))
        rouge_scores[f'{name}_salience_overlap_score_cand'].append(calculate_overlap_score(datum.ref, datum.salience, datum.cand))
        rouge_scores[f'{name}_salience_overlap_score_ref'].append(calculate_overlap_score(datum.ref, datum.salience, datum.ref))
        rouge_scores[f'{name}_no_salience_count'].append(count_salience(datum.salience)[0])
        rouge_scores[f'{name}_salience_count'].append(count_salience(datum.salience)[1])
        rouge_scores[f'{name}_salience_count_1'].append(count_salience(datum.salience)[2])
result = pd.DataFrame.from_dict(rouge_scores)
#%%
result.to_csv((output_folder / 'data.csv'), sep='\t', encoding='utf-8')
#%%
f = '_rL_fmeasure'
o = '_salience_overlap_score_cand'
l = '_length_cand'
result2 = {
    'clean': {
        'overlap': result[f'clean{o}'].mean(),
        'length': result[f'clean{l}'].mean(),
        'RL': result[f'clean{f}'].mean(),
    },
    'emb_mlp': {
        'overlap': result[f'emb_mlp_1{o}'].mean(),
        'length': result[f'emb_mlp_1{l}'].mean(),
        'RL': result[f'emb_mlp_1{f}'].mean(),
    },
    'salience_attn': {
        'overlap': result[f'salience_attn{o}'].mean(),
        'length': result[f'salience_attn{l}'].mean(),
        'RL': result[f'salience_attn{f}'].mean(),
    },
    'salience_emb_attn': {
        'overlap': result[f'salience_emb_attn{o}'].mean(),
        'length': result[f'salience_emb_attn{l}'].mean(),
        'RL': result[f'salience_emb_attn{f}'].mean(),
    },
    'gold': {
        'overlap': result[f'salience_emb_attn_revise_salience_overlap_score_ref'].mean(),
        'length': result[f'salience_emb_attn_revise{l}'].mean(),
        'RL': 0,
    }
}
#%%
result2 = pd.DataFrame.from_dict(result2)
result2.to_csv((output_folder / 'data2.csv'), sep=',')
#%%
great_salience_emb_mlp_1 = result[result['emb_mlp_1_rL_fmeasure'] > result['clean_rL_fmeasure']]
less_salience_emb_mlp_1 = result[result['emb_mlp_1_rL_fmeasure'] < result['clean_rL_fmeasure']]
great_salience_emb_mlp_2 = result[result['emb_mlp_2_rL_fmeasure'] > result['clean_rL_fmeasure']]
less_salience_emb_mlp_2 = result[result['emb_mlp_2_rL_fmeasure'] < result['clean_rL_fmeasure']]
print('Great salience overlap')
