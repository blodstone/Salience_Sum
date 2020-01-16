import argparse
from pathlib import Path


def main():
    src_path = Path(args.src)
    tgt_path = Path(args.tgt)
    output_folder = Path(args.output)
    print('Converting.')
    with src_path.open() as src_f, tgt_path.open() as tgt_f, (output_folder / f'{args.name}.txt').open('w') as o:
        results = [f'{s.strip()}\t{t.strip()}' for s, t in zip(src_f, tgt_f)]
        o.write('\n'.join(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='The source file.')
    parser.add_argument('-tgt', help='The target file.')
    parser.add_argument('-output', help='The output folder.')
    parser.add_argument('-name', help='The name')
    args = parser.parse_args()
    main()