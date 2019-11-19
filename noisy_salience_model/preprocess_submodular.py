import os
if __name__ == '__main__':
    file_path = '../data/bbc_allen/train.tsv'
    output = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    index = 0
    with open(file_path) as file:
        for line in file:
            index += 1
            print('Process line {}'.format(index))
            doc, summ = line.strip().split('\t')
            file_write = open(os.path.join(output, '{}.txt'.format(index)), 'w')
            file_write.write(doc)
            file_write.close()
