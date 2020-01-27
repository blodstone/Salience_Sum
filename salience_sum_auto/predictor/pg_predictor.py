from pathlib import Path

import numpy
import torch
from allennlp.common import JsonDict, Registrable
from allennlp.data import Instance, DatasetReader, Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.models import Model
from allennlp.nn import util
from allennlp.predictors import Predictor
from overrides import overrides
from tqdm import tqdm

from pointer_generator.reader import SummDataReader


class Seq2SeqPredictor:

    def __init__(self, model: Model,
                 data_reader: SummDataReader,
                 batch_size: int,
                 cuda_device: int):
        self.cuda_device = cuda_device
        self.iterator = BucketIterator(batch_size=batch_size,
                                       sorting_keys=[("source_tokens", "num_tokens")])
        self.model = model
        self.data_reader = data_reader

    def _extract_data(self, batch) -> numpy.ndarray:
        out_dict = self.model(**batch)
        return out_dict

    def predict(self, file_path: str, vocab_path: str):
        ds = self.data_reader.read(file_path)
        vocab = Vocabulary.from_files(vocab_path)
        self.iterator.index_with(vocab)
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return preds
