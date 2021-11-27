import spacy
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from gensim.utils import tokenize
from sklearn.manifold import TSNE
from umap import UMAP
import time


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
        self.spacy_model.max_length = 1250000

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

    def make_embeddings(self, tokens: list) -> list:
        embeddings = []
        for token in tokens:
            embeddings.append(self.spacy_model(token).vector)
        return embeddings

    @staticmethod
    def get_embeddings(embeds, u_s, s):
        res = []
        for el in s:
            ind = u_s.index(el)
            res.append(embeds[ind])
        return res

    @staticmethod
    def apply_tsne(embeddings):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=3000, metric='cosine', random_state=0)
        np.set_printoptions(suppress=True)
        y = tsne.fit_transform(embeddings)
        return y

    @staticmethod
    def apply_umap(embeddings):
        umap = UMAP(n_components=2, init='spectral', random_state=42)
        y = umap.fit_transform(embeddings)
        return y

    # produces outliers
    def foreground_transform(self, filtered_tokens_f_set, filtered_tokens_b_set, y_b):
        y_f = []
        for filtered_token_f_set in filtered_tokens_f_set:
            if filtered_token_f_set in filtered_tokens_b_set:
                i = filtered_tokens_b_set.index(filtered_token_f_set)
                y_f.append(y_b[i])
            else:
                el = self.apply_tsne(self.make_embeddings(filtered_token_f_set))
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

        return res

    @staticmethod
    def add_size(df):
        df['size'] = np.where(df['set'] == 'foreground', df['counts'], df['counts']+1)
        df.to_csv('res.csv')
        return df

    def stats(self, file):

        start = time.time()
        orig_text = self.read_text(file)
        end = time.time()
        print("4MB file reading time {:0.2f}".format(end-start))  # little

        start = time.time()
        self.set_spacy_model('en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0')
        end = time.time()
        print("model setting time {:0.2f}".format(end - start))  # 1.5 seconds

        limits = [10500, 57000, 175000, 450000, 1150000]

        results = {'words': [], 'unique_words': [], 'lemmatization': [], 'embeddings': [], 'umap': [], 'df': []}

        fig, axs = plt.subplots(2, 2)

        for limit in limits:
            print(limit)
            text = orig_text[:limit]

            start = time.time()
            lemmatized_text = self.lemmatize(text)
            end = time.time()
            # print("lemmatization time {:0.2f}".format(end-start))
            results['lemmatization'].append(end-start)

            # start = time.time()
            tokenized_text = self.tokenize(lemmatized_text)
            # end = time.time()
            # print("tokenization time {:0.2f}".format(end-start))  # little
            results['words'].append(len(tokenized_text))

            # start = time.time()
            filtered_text, filtered_text_set = self.remove_stopwords(tokenized_text)
            # end = time.time()
            # print("removing stopwords time {:0.2f}".format(end-start)) # little
            results['unique_words'].append(len(filtered_text_set))

            start = time.time()
            embeddings = self.make_embeddings(list(filtered_text_set))
            end = time.time()
            # print("making embeddings time {:0.2f}".format(end-start))
            results['embeddings'].append(end-start)

            start = time.time()
            y = self.apply_umap(embeddings)
            end = time.time()
            # print("UMAP time {:0.2f}".format(end-start))
            results['umap'].append(end-start)

            start = time.time()
            df = self.make_dataframe(filtered_text, filtered_text_set, np.array(y), 'foreground')
            end = time.time()
            # print("making dataframe time {:0.2f}".format(end-start))
            results['df'].append(end-start)

            # start = time.time()
            # df = self.add_size(df)
            # end = time.time()
            # print("adding size column time {:0.2f}".format(end-start)) # little
            #
            # start = time.time()
            # fig = self.plot_plotly(df)
            # end = time.time()
            # print("making plot time {:0.2f}".format(end-start)) # ~3 seconds
            #
            # # fig.show()

        axs[0, 0].plot(results['words'], results['lemmatization'], 'ro')
        axs[0, 0].set_title('Lemmatization time')
        axs[0, 0].set_xlabel('Number of words')
        axs[0, 0].set_ylabel('Time in seconds')

        axs[0, 1].plot(results['unique_words'], results['embeddings'], 'ro')
        axs[0, 1].set_title('Making embeddings time')
        axs[0, 1].set_xlabel('Number of unique words')
        axs[0, 1].set_ylabel('Time in seconds')

        axs[1, 0].plot(results['unique_words'], results['umap'], 'ro')
        axs[1, 0].set_title('UMAP time')
        axs[1, 0].set_xlabel('Number of unique words')
        axs[1, 0].set_ylabel('Time in seconds')

        axs[1, 1].plot(results['unique_words'], results['df'], 'ro')
        axs[1, 1].set_title('df time')
        axs[1, 1].set_xlabel('Number of unique words')
        axs[1, 1].set_ylabel('Time in seconds')

        fig.tight_layout()

        plt.show()
