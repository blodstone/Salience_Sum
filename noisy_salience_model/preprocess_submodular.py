import os

if __name__ == '__main__':
    folder = '/home/acp16hh/Projects/Research/Experiments/Exp_Elly_Human_Evaluation/src/Mock_Dataset_2/Hardy-Data-Final'
    output = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        doc = open(filepath).readlines()
        body = False
        doc_lines = []
        for line in doc:
            if line.startswith('[SN]RESTBODY[SN]'):
                body = True
                continue
            if body:
                doc_lines.append(line.strip())
        doc_lines = ' '.join(doc_lines)
        outfile = open(os.path.join(output, file.split('.')[0]+'.txt'), 'w')
        outfile.write(doc_lines)
        outfile.close()
