import argparse

from pointer_generator_salience import summarizer as pg_summarizer
from pg_salience_feature import summarizer as pg_sal_summarizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-module', help='Summarizer module.')
    parser.add_argument('-input', help='Input tsv.')
    parser.add_argument('-vocab_path', help='Vocabulary path.')
    parser.add_argument('-model', help='Model path.')
    parser.add_argument('-model_config', help='Model config.')
    parser.add_argument('-output_path', help='Output path of the prediction.')
    parser.add_argument('-batch_size', help='Batch size', type=int, default=24)
    parser.add_argument('--cuda', help='Use cuda.', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        cuda = 'cuda'
    else:
        cuda = 'cpu'
    if args.module == 'pointer_generator_salience':
        pg_summarizer.summarize(args.input, args.vocab_path, args.model,
                             args.model_config, args.output_path, args.batch_size, cuda)
    elif args.module == 'pg_salience_feature':
        pg_sal_summarizer.summarize(args.input, args.vocab_path, args.model,
                             args.model_config, args.output_path, args.batch_size, cuda)
