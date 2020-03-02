import os


def process_set(split, folder_path):
    file = os.path.join(folder_path, split + '.tsv.tagged')
    with open(file) as f:
        srcs = []
        tgts = []
        saliences = []
        for line in f:
            src_salience, tgt = line.strip().split('\t')
            src, salience = zip(*[(pair.split('|%|')[0], pair.split('|%|')[1]) for pair in src_salience.split()])
            srcs.append(' '.join(src))
            tgts.append(tgt)
            saliences.append(' '.join(salience))
        open(os.path.join(folder_path, split + '.src.txt'), 'w').write('\n'.join(srcs))
        open(os.path.join(folder_path, split + '.tgt.txt'), 'w').write('\n'.join(tgts))
        open(os.path.join(folder_path, split + '.sal.txt'), 'w').write('\n'.join(saliences))


def main():
    folder_path = '../data/bbc_highres'
    splits = ['test']
    for split in splits:
        process_set(split, folder_path)


if __name__ == '__main__':
    main()


