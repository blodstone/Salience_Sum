import os


def retrieve(doc_name, summ_name, doc_path, summ_path, output_path):
    contents = []
    doc_files = [(name, int(name.split('.')[0])) for name in os.listdir(doc_path)]
    doc_files.sort(key = lambda x: x[1])
    summ_files = [(name, int(name.split('.')[0])) for name in os.listdir(summ_path)]
    summ_files.sort(key = lambda x: x[1])
    for doc, summ in zip(doc_files, summ_files):
        doc_file = doc[0]
        summ_file = summ[0]
        if doc_file != summ_file:
            print('Error')
            break
        else:
            doc = open(os.path.join(doc_path, doc_file)).readlines()[0]
            summ_content = list(open(os.path.join(summ_path, summ_file)).readlines())
            if len(summ_content) == 0:
                print('Empty summary on file: {} of {}'.format(summ_file, summ_path))
                summ = ''
            else:
                summ = summ_content[0]
            content = '{}\t{}'.format(doc, summ)
        contents.append(content)
    output = open(os.path.join(output_path, 'gen.{}.{}.tsv'.format(doc_name, summ_name)), 'w')
    output.write('\n'.join(contents))


if __name__ == '__main__':
    doc_folder = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    docs_name = ['train', 'val']
    summs_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs/'
    summs_group = [['submodular', 'textrank', 'centroid'], ['submodular_val', 'textrank_val', 'centroid_val']]
    output_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen'
    for doc_name, summs_group in zip(docs_name, summs_group):
        for summ_name in summs_group:
            doc_path = os.path.join(doc_folder, doc_name)
            summ_path = os.path.join(summs_path, summ_name)
            retrieve(doc_name, summ_name, doc_path, summ_path, output_path)







