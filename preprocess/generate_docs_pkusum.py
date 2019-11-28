import os
import tqdm

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

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
    # i_paths = ['train.tsv', 'val.tsv']
    i_paths = ['test.tsv']
    g_paths = [['submodular', 'textrank', 'centroid']]
    # doc_names = ['train', 'val']
    doc_names = ['test']
    output = '../../PKUSUMSUM/summs'
    create_folder(output)
    output_doc = '../../PKUSUMSUM/docs'
    create_folder(output_doc)
    output_gold = '../../PKUSUMSUM/gold'
    create_folder(output_gold)
    # Generate file for PKUSUM
    #for i_path, o_paths in zip(i_paths, g_paths):
    #    for o_path in o_paths:
    #        input_path = os.path.join(file_path, i_path)
    #        output_path = os.path.join(output, o_path)
    #        create_folder(output_path)
    #        generate(input_path, output_path)
    # Generate docs and gold
    for i_path, doc_name in zip(i_paths, doc_names):
        input_path = os.path.join(file_path, i_path)
        # Generate doc
        output_path = os.path.join(output_doc, doc_name)
        create_folder(output_path)
        generate(input_path, output_path)
        # Generate gold
        output_path = os.path.join(output_gold, doc_name)
        create_folder(output_path)
        generate(input_path, output_path, 'summ')
