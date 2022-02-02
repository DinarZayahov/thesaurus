import spacy
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from gensim.utils import tokenize
from sklearn.manifold import TSNE
from umap import UMAP  # 0.4.2
import time
import pickle
import os
from collections import Counter
from minisom import MiniSom

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure


MAX_LENGTH = 1250000
LEMMATIZATION_THRESHOLD = 500000
MODEL = 'en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0'


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
        self.spacy_model.max_length = MAX_LENGTH

    def lemmatize(self, text, length):
        if length < LEMMATIZATION_THRESHOLD:
            doc = self.spacy_model(text)
            result = " ".join([token.lemma_ for token in doc])
            return result
        else:
            for doc in self.spacy_model.pipe([text], batch_size=32, n_process=3, disable=["parser", "ner"]):
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
        return filtered_tokens, list(dict.fromkeys(filtered_tokens))

    def make_embeddings(self, tokens: list) -> list:
        embeddings_filename = 'embeddings.pickle'
        if os.path.exists(embeddings_filename):
            print('Found cache..')
            embeddings_file = open(embeddings_filename, 'rb')
            changed = False
            dictionary = pickle.load(embeddings_file)
            result = []
            for token in tokens:
                if token in dictionary:
                    result.append(dictionary[token])
                else:
                    e = self.spacy_model(token).vector
                    dictionary[token] = e
                    changed = True
                    result.append(e)
            if changed:
                print('Rewriting cache..')
                embeddings_file.close()
                os.remove(embeddings_filename)
                new_embeddings_file = open(embeddings_filename, 'wb')
                pickle.dump(dictionary, new_embeddings_file)
            return result
        else:
            print('Cache not found..')
            dictionary = dict()
            for token in tokens:
                dictionary[token] = self.spacy_model(token).vector
            embeddings_file = open(embeddings_filename, 'wb')
            pickle.dump(dictionary, embeddings_file)
            return list(dictionary.values())

    @staticmethod
    def get_embeddings(embeds, u_s, s):
        res = []
        for el in s:
            ind = u_s.index(el)
            res.append(embeds[ind])
        return res

    @staticmethod
    def apply_tsne(embeddings):
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300, metric='manhattan', random_state=0)
        np.set_printoptions(suppress=True)
        y = tsne.fit_transform(embeddings)
        return y

    @staticmethod
    def apply_umap(embeddings):
        umap = UMAP(n_components=2, metric='euclidean', random_state=42, densmap=True)
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
        counter = Counter(filtered_tokens)
        tokens_list = filtered_tokens_set

        counts = []
        for token in tokens_list:
            c = counter[token]
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
        df['size'] = np.where(df['set'] == 'foreground', df['counts'], df['counts'])
        # df['size'] = np.where(df['set'] == 'foreground', df['counts'], df['counts']+1) # to make background larger
        df.to_csv('res.csv')
        return df

    @staticmethod
    def get_grid_size(n):
        neurons_num = 5*np.sqrt(n)
        return int(np.ceil(np.sqrt(neurons_num)))

    def plot_bokeh(self, embeddings_f, embeddings_b, filtered_ftext_set, filtered_btext_set):
        HEXAGON_SIZE = 54
        DOT_SIZE = 20

        GRID_SIZE = self.get_grid_size(len(embeddings_b))
        PLOT_SIZE = HEXAGON_SIZE * (GRID_SIZE + 1)

        som = MiniSom(GRID_SIZE, GRID_SIZE, np.array(embeddings_b).shape[1], sigma=5, learning_rate=.2,
                      activation_distance='euclidean', topology='hexagonal', neighborhood_function='bubble',
                      random_seed=10)

        som.train(embeddings_b, 1000, verbose=True)

        b_label = []

        b_weight_x, b_weight_y = [], []
        for cnt, i in enumerate(embeddings_b):
            w = som.winner(i)
            wx, wy = som.convert_map_to_euclidean(xy=w)
            wy = wy * np.sqrt(3) / 2
            b_weight_x.append(wx)
            b_weight_y.append(wy)
            b_label.append(filtered_btext_set[cnt])

        f_label = []

        f_weight_x, f_weight_y = [], []
        for cnt, i in enumerate(embeddings_f):
            w = som.winner(i)
            wx, wy = som.convert_map_to_euclidean(xy=w)
            wy = wy * np.sqrt(3) / 2
            f_weight_x.append(wx)
            f_weight_y.append(wy)
            f_label.append(filtered_ftext_set[cnt])

        # initialise figure/plot
        fig = figure(plot_height=PLOT_SIZE, plot_width=PLOT_SIZE,
                     match_aspect=True,
                     tools="pan")

        fig.axis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None

        # create data stream for plotting
        b_source_pages = ColumnDataSource(
            data=dict(
                wx=b_weight_x,
                wy=b_weight_y,
                species=b_label
            )
        )

        f_source_pages = ColumnDataSource(
            data=dict(
                wx=f_weight_x,
                wy=f_weight_y,
                species=f_label
            )
        )

        fig.hex(x='wy', y='wx', source=b_source_pages,
                fill_alpha=1.0, line_alpha=1.0,
                size=HEXAGON_SIZE)

        fig.scatter(x='wy', y='wx', source=f_source_pages,
                    fill_color='orange',
                    size=DOT_SIZE)

        TOOLTIPS = """
            <div style ="border-style: solid;border-width: 15px;background-color:black;">         
                <div>
                    <span style="font-size: 12px; color: white;font-family:century gothic;"> @species</span>
                </div>
            </div>
            """

        # add hover-over tooltip
        fig.add_tools(HoverTool(
            tooltips=[
                ("label", '@species')],
            # tooltips=TOOLTIPS,
            mode="mouse",
            point_policy="follow_mouse"
        ))

        return fig

    # one graph experiment
    def stats1(self, file):

        start = time.time()
        orig_text = self.read_text(file)
        end = time.time()
        print("4MB file reading time {:0.2f}".format(end-start))  # little

        start = time.time()
        self.set_spacy_model(MODEL)
        end = time.time()
        print("model setting time {:0.2f}".format(end - start))  # 1.5 seconds

        limits = [10500, 57000, 175000, 450000, 1150000]
        # 1k, 5k, 15k, 25k, 45k, 90k

        limit = limits[0]
        text = orig_text[:limit]
        print(len(text))

        start = time.time()
        lemmatized_text = self.lemmatize(text, len(text))
        end = time.time()
        print("lemmatization time {:0.2f}".format(end-start))

        start = time.time()
        tokenized_text = self.tokenize(lemmatized_text)
        end = time.time()
        print("tokenization time {:0.2f}".format(end-start))  # little

        start = time.time()
        filtered_text, filtered_text_set = self.remove_stopwords(tokenized_text)
        end = time.time()
        print("removing stopwords time {:0.2f}".format(end-start))  # little

        fw = open('words.pickle', 'wb')
        pickle.dump(filtered_text_set, fw)

        start = time.time()
        embeddings = self.make_embeddings(filtered_text_set)
        end = time.time()
        print("making embeddings time {:0.2f}".format(end-start))

        HEXAGON_SIZE = 54
        DOT_SIZE = 20

        GRID_SIZE = self.get_grid_size(len(embeddings))
        PLOT_SIZE = HEXAGON_SIZE * (GRID_SIZE+1)

        som = MiniSom(GRID_SIZE, GRID_SIZE, np.array(embeddings).shape[1], sigma=5, learning_rate=.2,
                      activation_distance='euclidean', topology='hexagonal', neighborhood_function='bubble',
                      random_seed=10)

        som.train(embeddings, 1000, verbose=True)

        label = []

        weight_x, weight_y = [], []
        for cnt, i in enumerate(embeddings):
            w = som.winner(i)
            wx, wy = som.convert_map_to_euclidean(xy=w)
            wy = wy * np.sqrt(3) / 2
            weight_x.append(wx)
            weight_y.append(wy)
            label.append(filtered_text_set[cnt])

        # initialise figure/plot
        fig = figure(plot_height=PLOT_SIZE, plot_width=PLOT_SIZE,
                     match_aspect=True,
                     tools="pan")

        fig.axis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None

        # create data stream for plotting
        source_pages = ColumnDataSource(
            data=dict(
                wx=weight_x,
                wy=weight_y,
                species=label
            )
        )

        fig.hex(x='wy', y='wx', source=source_pages,
                fill_alpha=1.0, line_alpha=1.0,
                size=HEXAGON_SIZE)

        fig.scatter(x='wy', y='wx', source=source_pages,
                    fill_color='orange',
                    size=DOT_SIZE)

        # add hover-over tooltip
        fig.add_tools(HoverTool(
            tooltips=[
                ("label", '@species')],
            mode="mouse",
            point_policy="follow_mouse"
        ))

        return fig

    # for time-statistics
    def stats2(self, file):

        start = time.time()
        orig_text = self.read_text(file)
        end = time.time()
        print("4MB file reading time {:0.2f}".format(end - start))  # little

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
            lemmatized_text = self.lemmatize(text, len(text))
            end = time.time()
            results['lemmatization'].append(end - start)

            # start = time.time()
            tokenized_text = self.tokenize(lemmatized_text)
            # end = time.time()
            results['words'].append(len(tokenized_text))

            # start = time.time()
            filtered_text, filtered_text_set = self.remove_stopwords(tokenized_text)
            # end = time.time()
            results['unique_words'].append(len(filtered_text_set))

            start = time.time()
            embeddings = self.make_embeddings(filtered_text_set)
            end = time.time()
            results['embeddings'].append(end - start)

            start = time.time()
            y = self.apply_umap(embeddings)
            end = time.time()
            results['umap'].append(end - start)

            start = time.time()
            _ = self.make_dataframe(filtered_text, filtered_text_set, np.array(y), 'foreground')
            end = time.time()
            results['df'].append(end - start)

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

    # to experiment with UMAP-parameters
    def stats3(self, file):

        # array = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'mahalanobis',
        #          'wminkowski', 'cosine', 'correlation', 'hamming', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto',
        #          'sokalmichener', 'sokalsneath', 'yule']

        array = [2, 5, 10, 20, 50, 100, 200, 400]
        new_array = array
        n = len(new_array)

        start = time.time()
        orig_text = self.read_text(file)
        end = time.time()
        print("4MB file reading time {:0.2f}".format(end - start))  # little

        start = time.time()
        self.set_spacy_model('en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0')
        end = time.time()
        print("model setting time {:0.2f}".format(end - start))  # 1.5 seconds

        limits = [10500, 57000, 175000, 450000, 1150000]
        # 1k, 5k, 15k, 25k, 45k, 90k

        limit = limits[0]
        text = orig_text[:limit]

        figs = []

        for value in new_array:

            try:
                start = time.time()
                lemmatized_text = self.lemmatize(text, len(text))
                end = time.time()
                print("lemmatization time {:0.2f}".format(end-start))

                start = time.time()
                tokenized_text = self.tokenize(lemmatized_text)
                end = time.time()
                print("tokenization time {:0.2f}".format(end-start))  # little

                start = time.time()
                filtered_text, filtered_text_set = self.remove_stopwords(tokenized_text)
                end = time.time()
                print("removing stopwords time {:0.2f}".format(end-start))  # little

                start = time.time()
                embeddings = self.make_embeddings(filtered_text_set)
                end = time.time()
                print("making embeddings time {:0.2f}".format(end-start))

                start = time.time()
                y = self.apply_umap(embeddings)
                # y = self.apply_tsne(embeddings)
                end = time.time()
                print("UMAP time {:0.2f}".format(end-start))
                # print("t-SNE time {:0.2f}".format(end - start))

                start = time.time()
                df = self.make_dataframe(filtered_text, filtered_text_set, np.array(y), 'foreground')
                end = time.time()
                print("making dataframe time {:0.2f}".format(end-start))

                start = time.time()
                df = self.add_size(df)
                end = time.time()
                print("adding size column time {:0.2f}".format(end-start))  # little

                start = time.time()
                fig = self.plot_plotly(df)
                end = time.time()
                print("making plot time {:0.2f}".format(end-start))  # ~3 seconds

                figs.append(fig)
                # fig.show()
            except:
                print('error in', value)

        figs_traces = []
        for f in figs:
            a = []
            for trace in range(len(f["data"])):
                a.append(f["data"][trace])
            figs_traces.append(a)

        half = round(n/2)
        figure = make_subplots(rows=round(n/2), cols=2)
        for i in range(len(figs_traces)):
            for traces in figs_traces[i]:
                if i+1 <= half:
                    row = i+1
                    col = 1
                else:
                    row = i+1-half
                    col = 2
                figure.append_trace(traces, row=row, col=col)

        figure.show()
