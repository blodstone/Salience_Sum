"""
This is just for adding noisy dummy data
"""
import numpy as np

def gen_lines(path):
    with open(path) as file:
        new_lines = []
        for line in file:
            src, tgt = line.strip().split('\t')
            words = src.split()
            alphas = np.random.beta(a=2, b=5, size=len(words))
            probs = np.random.dirichlet(alphas, 1)
            comb = []
            for word, prob in zip(words, np.squeeze(probs, axis=0).tolist()):
                if prob == 0.0:
                    print('found')
                comb.append('{}###{:.6f}'.format(word, prob))
            new_lines.append(' '.join(comb) + '\t' + tgt + '\n')
        return new_lines

train_path = '../data/bbc/train.tsv'
val_path = '../data/bbc/val.tsv'
tagged_train_path = '../data/bbc/train.tsv.tagged'
tagged_val_path = '../data/bbc/val.tsv.tagged'
tagged_train_file = open(tagged_train_path, 'w')
tagged_val_file = open(tagged_val_path, 'w')
tagged_train_file.writelines(gen_lines(train_path))
tagged_val_file.writelines(gen_lines(val_path))
tagged_val_file.close()
tagged_train_file.close()
