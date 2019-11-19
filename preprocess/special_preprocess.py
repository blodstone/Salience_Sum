import os

if __name__ == '__main__':
    doc_folder = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    summ_submodular_folder = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs/textrank'
    docs = []
    summs = []
    filenames = []
    for filename in os.listdir(doc_folder):
        doc = ' '.join(open(os.path.join(doc_folder, filename)).readlines())
        summ = ' '.join([line.strip() for line in open(os.path.join(summ_submodular_folder, filename)).readlines()])
        docs.append(doc)
        summs.append(summ)
        filenames.append(filename.split('.')[0])
    final_doc = open('sample_data/train_src.txt', 'w')
    final_summ = open('sample_data/textrank_train_tgt.txt', 'w')
    final_doc_id = open('sample_data/doc_id.txt', 'w')
    final_doc.write('\n'.join(docs))
    final_doc_id.write('\n'.join(filenames))
    final_summ.write('\n'.join(summs))






