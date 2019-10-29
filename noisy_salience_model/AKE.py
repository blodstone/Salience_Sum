import spacy


def filter_words(doc):
    result = []
    for word in doc:
        if word.tag == 'NOUN' or word.tag == 'ADJ' or word.tag == 'VERB':
            result += word.text
    return result


def parse_graf():
    pass


def build_graph(words, window):
    parse_graf()


def run(max_words, src):
    nlp = spacy.load('en_core_web_sm')
    docs = [filter_words(doc) for doc in list(nlp.pipe(src))]
    for doc in docs:
        build_graph(doc)
