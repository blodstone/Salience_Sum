import argparse
import json
from pathlib import Path


def main():
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    # Hardcoded train, test and validation name
    train_path = input_path / 'train.tsv'
    test_path = input_path / 'test.tsv'
    validation_path = input_path / 'validation.tsv'
    index_json = {
        'train': [],
        'test': [],
        'validation': [],
    }
    i, index_json = split_file(train_path, 'train', 0, index_json, output_path)
    i, index_json = split_file(test_path, 'test', 0, index_json, output_path)
    _, index_json = split_file(validation_path, 'validation', 0, index_json, output_path)
    json.dump(index_json, (output_path / 'CNN-TRAINING-DEV-TEST-SPLIT.json').open('w'))


def split_file(path, name, i, index_json, output_path):
    docs_path = output_path / 'docs'
    summaries_path = output_path / 'summaries'
    docs_path.mkdir(exist_ok=True, parents=True)
    summaries_path.mkdir(exist_ok=True, parents=True)
    for line in path.open().readlines():
        if line.strip() == '':
            continue
        src, tgt = line.split('\t')
        index_json[name].append(i)
        if not args.json_only:
            (docs_path / f'{str(i)}.txt').open('w').write(src)
            (summaries_path / f'{str(i)}.txt').open('w').write(tgt)
        i += 1
    return i, index_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', help='The path containing test, train and validation.')
    parser.add_argument('-output_path', help='The output path.')
    parser.add_argument('--json_only', help='Produce json only.', action='store_true')
    args = parser.parse_args()
    main()
