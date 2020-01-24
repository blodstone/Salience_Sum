import argparse
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model

from pointer_generator_salience import SummDataReader, Seq2SeqPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Input tsv.')
    parser.add_argument('-vocab_path', help='Vocabulary path.')
    parser.add_argument('-model', help='Model path.')
    parser.add_argument('-model_config', help='Model config.')
    parser.add_argument('--cuda', help='Use cuda.', action='store_true')
    args = parser.parse_args()
    reader = SummDataReader(predict=True, source_max_tokens=400, use_salience=False)
    config = Params.from_file(args.model_config)
    if args.cuda:
        model_state = torch.load(args.model, map_location=torch.device(0))
    else:
        model_state = torch.load(args.model, map_location=torch.device('cpu'))
    vocab = Vocabulary.from_files(args.vocab_path)
    model = Model.from_params(vocab=vocab, params=config.get('model'))
    model.load_state_dict(model_state)
    predictor = Seq2SeqPredictor(model=model, data_reader=reader, batch_size=24, cuda_device=-1)
    output = predictor.predict(file_path=args.input, vocab_path=args.vocab_path)
    for out in output:
        predictions = out['results']
        for sent in predictions['predictions']:
            print(' '.join([vocab.get_index_to_token_vocabulary()[idx.item()] for idx in sent[0]]))
