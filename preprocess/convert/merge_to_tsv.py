"""
Take a src and tgt files (standard input in opennmt and fairseq) and join them into a single tsv
"""
import json
import argparse
from pathlib import Path


def main():
    raw_docs = Path(args.raw_docs)
    raw_summs = Path(args.raw_summs)
    index = json.load(open(args.index))
    reverse_index = {doc_id: dataset for dataset, doc_ids in index.items() for doc_id in doc_ids}
    output_path = Path(args.output_path)

    output_text = {
        'train': [],
        'test': [],
        'validation': []
    }
    for i, raw_doc in enumerate(raw_docs.iterdir()):
        doc_id = raw_doc.stem
        if doc_id in reverse_index:
            print(f'{doc_id} ({i})')
            summary_text = ' '.join(
                [line.strip().lower() for line in (raw_summs / f'{doc_id}.fs').open().readlines()])
            text = ' '.join([line.strip().lower() for line in raw_doc.open().readlines()])
            tsv = f'{text}\t{summary_text}'
            dataset = reverse_index[doc_id]
            output_text[dataset].append(tsv)

    for dataset, text_list in output_text.items():
        (output_path / f'{dataset}.tsv').write_text('\n'.join(text_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_docs', help='The raw documents of BBC.')
    parser.add_argument('-raw_summs', help='The raw summaries of BBC.')
    parser.add_argument('-index', help='The index for splitting raw document.')
    parser.add_argument('-output_path', help='The output path.')
    args = parser.parse_args()
    main()
