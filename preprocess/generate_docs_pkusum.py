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
    file_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/dev_bbc'
    i_paths = ['train.dev.tsv', 'val.dev.tsv']
    g_paths = [['submodular', 'textrank', 'centroid'], ['submodular_val', 'textrank_val', 'centroid_val']]
    doc_names = ['train', 'val']
    output = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs'
    output_doc = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    output_gold = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/gold'
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
