import argparse
import os
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from pg_salience_feature import SummDataReader, Seq2SeqPredictor


def summarize(input, vocab_path, model, model_config, output_path, batch_size, cuda, use_salience=True, use_constraint=False):
    input_file = Path(input)
    output_file = Path(output_path)
    if output_file.exists():
        print('Output file already exists. Deleting it.')
        os.remove(str(output_file))
    reader = SummDataReader(predict=True,
                            source_max_tokens=400,
                            use_salience=use_salience,
                            use_constraint=use_constraint)
    config = Params.from_file(model_config)
    model_state = torch.load(model, map_location=torch.device(cuda))
    vocab = Vocabulary.from_files(vocab_path)
    model = Model.from_params(vocab=vocab, params=config.get('model'))
    if cuda == 'cpu':
        cuda = -1
        model.cpu()
    else:
        cuda = 0
        model.cuda(cuda)
    model.load_state_dict(model_state)

    predictor = Seq2SeqPredictor(model=model, data_reader=reader, batch_size=batch_size, cuda_device=cuda)
    output = predictor.predict(file_path=str(input_file), vocab=vocab)
    write_constraints = []
    for out in output:
        results = out['results']
        if 'word_constraints' in out:
            constraints = out['word_constraints']
            if (output_file.parent / 'constraint.txt').exists():
                os.remove(str((output_file.parent / 'constraint.txt')))
            for constraint in constraints:
                write_constraints.append(str(constraint))
        for i, sent_idx in enumerate(range(len(results['predictions']))):
            out = ' '.join(
                 [token for token in results['predictions'][sent_idx] if token != END_SYMBOL and token != START_SYMBOL and token != '@@PADDING@@'])
            if i < len(results['predictions']):
                out = f'{out}\n'
            output_file.open('a').write(out)
    if len(write_constraints) > 0:
        (output_file.parent / 'constraint.txt').open('w').write('\n'.join(write_constraints))
