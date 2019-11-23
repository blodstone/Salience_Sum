import os
import tqdm

def generate(input_path, output_path, type='doc'):
    index = 0
    with open(input_path) as file:
        for line in tqdm.tqdm(file):
            index += 1
            doc, summ = line.strip().split('\t')
            file_write = open(os.path.join(output_path, '{}.txt'.format(index)), 'w')
            if type == 'doc':
                file_write.write(doc)
            else:
                file_write.write(summ)
            file_write.close()


if __name__ == '__main__':
    file_path = '../data/bbc_allen'
    i_paths = ['train.tsv', 'val.tsv']
    g_paths = [['submodular', 'textrank', 'centroid'], ['submodular_val', 'textrank_val', 'centroid_val']]
    doc_names = ['train', 'val']
    output = '../../PKUSUMSUM/summs'
    output_doc = '../../PKUSUMSUM/docs'
    output_gold = '../../PKUSUMSUM/gold'
    # for i_path, o_paths in zip(i_paths, g_paths):
    #     for o_path in o_paths:
    #         input_path = os.path.join(file_path, i_path)
    #         output_path = os.path.join(output, o_path)
    #         if not os.path.exists(output_path):
    #             os.mkdir(output_path)
    #         generate(input_path, output_path)
    for i_path, doc_name in zip(i_paths, doc_names):
        input_path = os.path.join(file_path, i_path)
        output_path = os.path.join(output_doc, doc_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        generate(input_path, output_path)
        output_path = os.path.join(output_gold, doc_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        generate(input_path, output_path, 'summ')
