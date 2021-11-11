import spacy
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from gensim.utils import tokenize
from sklearn.manifold import TSNE


class Thesaurus:
    def __init__(self):
        self.spacy_model = None

    @staticmethod
    def read_text(file):
        lines = []
        for line in file:
            line = line.decode('utf-8', 'ignore')
            lines.append(line)
        return ''.join(lines)

    def set_spacy_model(self, model):
        self.spacy_model = spacy.load(model)

    def lemmatize(self, text):
        doc = self.spacy_model(text)
        result = " ".join([token.lemma_ for token in doc])
        return result

    @staticmethod
    def tokenize(text):
        tokens = list(tokenize(text, to_lower=True))
        return tokens

    @staticmethod
    def get_stopwords(path):
        stopwords_file = open(path, 'r')
        stopwords = []
        for line in stopwords_file:
            stopwords.append(line[:-1])
        return stopwords

    def remove_stopwords(self, tokens: list):
        stopwords = self.get_stopwords('../data/extended_stopwords.txt')
        filtered_tokens = []
        for token in tokens:
            if token not in stopwords:
                filtered_tokens.append(token)
        return filtered_tokens, list(set(filtered_tokens))

    def get_embeddings(self, tokens: list) -> list:
        embeddings = []
        for token in tokens:
            embeddings.append(self.spacy_model(token).vector)
        return embeddings

    @staticmethod
    def apply_tsne(embeddings):
        tsne = TSNE(random_state=0)
        np.set_printoptions(suppress=True)
        y = tsne.fit_transform(embeddings)
        return y

    # might be done better
    def foreground_transform(self, filtered_tokens_f_set, filtered_tokens_b_set, y_b):
        y_f = []
        for filtered_token_f_set in filtered_tokens_f_set:
            if filtered_token_f_set in filtered_tokens_b_set:
                i = filtered_tokens_b_set.index(filtered_token_f_set)
                y_f.append(y_b[i])
            else:
                el = self.apply_tsne(self.get_embeddings(filtered_token_f_set))
                y_f.append(el[0])
        return y_f

    @staticmethod
    def plot_pyplot(y1, y2, ft2):
        plt.scatter(y1[:, 0], y1[:, 1], c='orange')

        plt.scatter(y2[:, 0], y2[:, 1], c='blue')
        for label, x, y in zip(ft2, y2[:, 0], y2[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()

    @staticmethod
    def plot_plotly(df):
        fig = px.scatter(df, x="x", y="y", color='set', size='size',
                         hover_data={'x': False, 'y': False,
                                     'words': True, 'counts': True, 'set': False, 'size': False},
                         color_discrete_sequence=["orange", "blue"])
        return fig

    @staticmethod
    def make_dataframe(filtered_tokens, filtered_tokens_set, y, type_):
        tokens_list = list(filtered_tokens_set)

        counts = []
        for token in tokens_list:
            c = filtered_tokens.count(token)
            counts.append(c)

        d = {'x': y[:, 0], 'y': y[:, 1], 'words': tokens_list, 'counts': counts, 'set': type_}

        df = pd.DataFrame(data=d)

        return df

    @staticmethod
    def join_dataframes(df1, df2):
        res = df1.append(df2, ignore_index=True)
        res['size'] = np.where(res['set'] == 'foreground', res['counts'], res['counts']+1)
        res.to_csv('res.csv')
        return res
