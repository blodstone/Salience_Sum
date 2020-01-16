with open('../../../data/bbc_allen_new/train.tsv.tagged') as val:
    new_line = []
    for line in val:
        src, target = line.strip().split('\t')
        target = target[2:-4].lower()
        new_line.append('{}\t{}\n'.format(src, target))
open('../../../data/bbc_allen_new/train.tsv.tagged', 'w').writelines(new_line)
