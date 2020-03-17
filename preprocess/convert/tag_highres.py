real = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc/ready/test.salience.tsv'
highres = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc/ready/highres/test.tsv'

with open(real) as file, open(highres) as file_2:
    tgt_seqs = []
    for line in file_2:
        line = line.strip()
        if line == '':
            continue
        src_seq, tgt_seq = line.split('\t')
        tgt_seqs.append(tgt_seq.strip().lower())
    reals = []
    lines = []
    for line in file:
        line = line.strip()
        if line == '':
            continue
        _, tgt_seq = line.split('\t')
        if tgt_seq.strip().lower() in tgt_seqs:
            lines.append(line)
    assert len(lines) == len(tgt_seqs)
    open('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc/ready/highres/test.salience.tsv', 'w').write('\n'.join(lines))
