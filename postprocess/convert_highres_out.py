#
# Convert tsv to highres input format (for weighted rouge)
#

from pathlib import Path

src = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc/highres/test.salience.tsv'
summ = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc/highres/Summaries'
output = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc/highres/output'
with open(src) as file:
    tgt_seqs = []
    for line in file:
        line = line.strip()
        if line == '':
            continue
        _, tgt_seq = line.split('\t')
        tgt_seqs.append(tgt_seq.strip().lower())
    filenames = []
    for tgt in tgt_seqs:
        for file in Path(summ).iterdir():
            s = file.read_text().strip().lower()
            if tgt == s:
                filenames.append(file.stem)
                break
    assert len(filenames) == len(tgt_seqs)

for file in Path(output).iterdir():
    (Path(output) / f'system_{file.stem}').mkdir(exist_ok=True, parents=True)
    save = (Path(output) / f'system_{file.stem}')
    for i, line in enumerate(file.open('r')):
        (save / f'{filenames[i]}.data').open('w').write(line)
