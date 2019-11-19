import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_scores_from_pickle():
    file = open('sample_data/df_scores.pickle', 'rb')
    return pickle.load(file)

def load_gold_scores_from_pickle():
    file = open('sample_data/df_gold_scores.pickle', 'rb')
    return pickle.load(file)

def showSingleAttention(tokens, gold, unsupervised, name):
    labels = tokens

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(80, 16))
    rects1 = ax.bar(x - width / 2, gold, width, label='Gold')
    rects2 = ax.bar(x + width / 2, unsupervised, width, label='Unsupervised')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by gold and unsupervised')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation='vertical')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # plt.show()
    fig.savefig('sample_data/' + name + '.png')

if __name__ == '__main__':
    chosen = 3
    df_scores_unsupervised = load_scores_from_pickle()
    df_scores_gold = load_gold_scores_from_pickle()
    unsupervised = list(df_scores_unsupervised[chosen])
    unsupervised = [round(number*100, 2) for number in unsupervised]
    gold = df_scores_gold[chosen][0].values.tolist()
    gold = [round(number*100, 2) for number in gold]
    src_path = 'sample_data/train_src.pickle'
    docs = pickle.load(open(src_path, 'rb'))
    tokens = [token.text for token in docs[chosen]]
    showSingleAttention(tokens, gold, unsupervised, 'Result')
