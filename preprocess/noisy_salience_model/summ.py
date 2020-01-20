from pathlib import Path
from difflib import SequenceMatcher
from noisy_salience_model.salience_model import Instance, SalienceSet, Salience


def process(salience_instance: Instance, summ_path: Path = None) -> Salience:
    summ_content = summ_path.open().readlines()
    salience = SalienceSet.init_salience_set(salience_instance.doc_size)
    document = ''.join([' '.join([word for word in line]) for line in salience_instance.doc])
    seq_matcher = SequenceMatcher(lambda x: x == ' ', b=document, autojunk=False)
    for summ_line in summ_content:
        summary = summ_line.strip().lower()
        seq_matcher.set_seq1(summary)
        blocks = seq_matcher.get_matching_blocks()
        for match in blocks:
            if match.size < 3:
                continue
            char_idx = match.b
            while char_idx < (match.b + match.size):
                line_idx, word_idx = salience_instance.index_map[char_idx]
                word = salience_instance.doc[line_idx][word_idx]
                salience[line_idx][word_idx] += 1.0
                char_idx += len(word) + 1
    return salience
