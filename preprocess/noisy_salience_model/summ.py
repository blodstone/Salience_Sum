from pathlib import Path
from difflib import SequenceMatcher
from noisy_salience_model.salience_model import Instance, SalienceSet, Salience
from nltk.tokenize import word_tokenize


def process(salience_instance: Instance, summ_path: Path = None) -> Salience:
    summ_content = summ_path.open().readlines()
    salience = SalienceSet.init_salience_set(salience_instance.doc_size)
    document = [word for line in salience_instance.doc for word in line]
    seq_matcher = SequenceMatcher(None, b=document, autojunk=False)
    for summ_line in summ_content:
        summary = word_tokenize(summ_line.strip().lower())
        seq_matcher.set_seq1(summary)
        blocks = seq_matcher.get_matching_blocks()
        for match in blocks:
            if match.size != 0:
                abs_idx = match.b
                while abs_idx < (match.b + match.size):
                    line_idx, word_idx = salience_instance.index_map[abs_idx]
                    word = salience_instance.doc[line_idx][word_idx]
                    assert word == document[abs_idx]
                    salience[line_idx][word_idx] += 1.0
                    abs_idx += 1
    return salience
