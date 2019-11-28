import os
from tqdm import tqdm

def process_set(split, folder_path):
    file = os.path.join(folder_path, split + '.tsv.tagged')
    with open(file) as f:
        srcs = []
        tgts = []
        saliences = []
        for line in tqdm(f):
            src_salience, tgt = line.strip().split('\t')
            src, salience = zip(*[(pair.split('|%|')[0], pair.split('|%|')[1]) for pair in src_salience.split()
                                  if pair.split('|%|')[0].strip() != ''])
            srcs.append(' '.join(src))
            tgts.append(tgt)
            saliences.append(' '.join(salience))
        salience_line = '\n'.join(saliences)
        src_line = '\n'.join(srcs)
        open(os.path.join(folder_path, split + '.src'), 'w').write(src_line)
        open(os.path.join(folder_path, split + '.tgt'), 'w').write('\n'.join(tgts))
        open(os.path.join(folder_path, split + '.sal'), 'w').write(salience_line)



def main():
    folder_path = '../data/bbc_allen'
    splits = ['train', 'val', 'test']
    for split in splits:
        print('Processing {}'.format(split))
        process_set(split, folder_path)

if __name__ == '__main__':
    main()


