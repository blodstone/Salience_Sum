import pickle
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


def get_all_space_indexes(astring):
    l = -1
    indexes = []
    while True:
        l = astring.find(" ", l + 1)
        if l == -1:
            break
        indexes.append(l)
    return indexes

def insert_str(string, index):
    return string[:index] + [' '] + string[index:]

def run(src_path, highlight_path, doc_id_path):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    docs = pickle.load(open(src_path, 'rb'))
    df_annotations = pickle.load(open(highlight_path, 'rb'))
    doc_ids = list(open(doc_id_path).readlines())
    result_labels = []
    for i, doc_id in enumerate(doc_ids):
        result_label = []
        df_doc = df_annotations[df_annotations.doc_id == doc_id.strip()]
        texts = list(df_doc['texts'])
        doc_space_indexes = get_all_space_indexes(' '.join([token.text for token in docs[i]]))
        doc_text_no_space = ' '.join([token.text for token in docs[i]]).replace(' ', '')
        temp_result = [0 for c in doc_text_no_space]
        for text in texts:
            atext = text.replace(' ', '')
            start = doc_text_no_space.find(atext)
            temp_result = [c+1 if i >= start and i <len(atext)+start else c for i, c in enumerate(temp_result)]
        new_result = temp_result
        for idx in doc_space_indexes:
            new_result = insert_str(new_result, idx)
        is_start = True
        for c in new_result:
            if is_start:
                if c == ' ':
                    result_label.append(0)
                else:
                    result_label.append(c)
                is_start = False
            if c == ' ':
                is_start = True
        result_labels.append(result_label)
    return result_labels


if __name__ == '__main__':
    src_path = '../sample_data/train_src.pickle'
    highlight_path = '../sample_data/df_gold.pickle'
    doc_id_path = '../sample_data/doc_id.txt'
    run(src_path, highlight_path, doc_id_path)
