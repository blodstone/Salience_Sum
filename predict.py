import os

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from salience_sum_old.predictor import Predictor
from salience_sum_old.reader.summ_data_reader import SummDataReader

vocab_path = 'data/dev_bbc/vocab'
# iterate over the dataset without changing its order
if os.path.exists(vocab_path):
    vocab = Vocabulary.from_files(vocab_path)
with open('data', 'wb') as f:
    model = torch.load(f)
iterator = BucketIterator(batch_size=4, sorting_keys=[("source_tokens", "num_tokens")])
iterator.index_with(vocab)

USE_GPU = False
predictor = Predictor(model, iterator, cuda_device=0 if USE_GPU else -1)
tokenizer = WordTokenizer(JustSpacesWordSplitter())
reader = SummDataReader(tokenizer, source_max_tokens=400, lazy=False)
test_ds = reader.read('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_highres/test.tsv.tagged')
test_preds = predictor.predict(test_ds)
