import spacy
import numpy as np
import matplotlib.pyplot as plt
from gensim.utils import tokenize
from sklearn.manifold import TSNE


class Thesaurus:
    def __init__(self):
        self.spacy_model = None

    @staticmethod
    def read_text(text_corpus):
        return ''.join(str(x) for x in text_corpus.read().splitlines())

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

    def remove_stopwords(self, tokens: list) -> set:
        stopwords = self.get_stopwords('../data/extended_stopwords.txt')
        filtered_tokens = []
        for token in tokens:
            if token not in stopwords:
                filtered_tokens.append(token)
        filtered_tokens = set(filtered_tokens)
        return filtered_tokens

    def get_embeddings(self, tokens: set) -> list:
        embeddings = []
        for token in tokens:
            embeddings.append(self.spacy_model(token).vector)
        return embeddings

    @staticmethod
    def apply_tsne(embeddings):
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        y = tsne.fit_transform(embeddings)
        return y

    @staticmethod
    def plot(y1, y2, ft2):
        plt.scatter(y1[:, 0], y1[:, 1], c='orange')

        plt.scatter(y2[:, 0], y2[:, 1], c='blue')
        for label, x, y in zip(ft2, y2[:, 0], y2[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()
