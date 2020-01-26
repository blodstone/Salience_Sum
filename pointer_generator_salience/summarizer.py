import argparse
import os
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from reader.summ_reader import SummDataReader
from predictor.pg_predictor import Seq2SeqPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Input tsv.')
    parser.add_argument('-vocab_path', help='Vocabulary path.')
    parser.add_argument('-model', help='Model path.')
    parser.add_argument('-model_config', help='Model config.')
    parser.add_argument('-output_path', help='Output path of the prediction.')
    parser.add_argument('-batch_size', help='Batch size', type=int, default=24)
    parser.add_argument('--cuda', help='Use cuda.', action='store_true')
    args = parser.parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output_path) / f'{input_file.stem}.out'
    if output_file.exists():
        print('Output file already exists. Deleting it.')
        os.remove(str(output_file))
    reader = SummDataReader(predict=True, source_max_tokens=400, use_salience=False)
    config = Params.from_file(args.model_config)
    if args.cuda:
        model_state = torch.load(args.model, map_location=torch.device(0))
    else:
        model_state = torch.load(args.model, map_location=torch.device('cpu'))
    vocab = Vocabulary.from_files(args.vocab_path)
    model = Model.from_params(vocab=vocab, params=config.get('model'))
    model.load_state_dict(model_state)
    predictor = Seq2SeqPredictor(model=model, data_reader=reader, batch_size=args.batch_size, cuda_device=-1)
    output = predictor.predict(file_path=str(input_file), vocab=vocab)
    for out in output:
        results = out['results']
        for i, sent_idx in enumerate(range(len(results['predictions']))):
            out = ' '.join([token for token in results['predictions'][sent_idx] if token != END_SYMBOL and token != START_SYMBOL])
            if i < len(results['predictions']):
                out = f'{out}\n'
            output_file.open('a').write(out)
