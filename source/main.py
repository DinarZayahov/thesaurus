from downloads import download_model
import spacy
from gensim.utils import tokenize
import streamlit as st
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Welcome to Thesaurus Visualization")
st.write(" ------ ")


def func(text):
    text = ''.join(str(x) for x in text.read().splitlines())

    nlp = spacy.load('en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0')
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])

    tokens = list(tokenize(text, to_lower=True))

    # Open file with stopwords
    stopwords_file = open('../data/extended_stopwords.txt', 'r')
    # Initialize empty list
    stopwords = []
    # Add stopwords to list
    for line in stopwords_file:
        stopwords.append(line[:-1])

    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)

    filtered_tokens = set(filtered_tokens)

    embeddings = []

    for token in filtered_tokens:
        embeddings.append(nlp(token).vector)

    return filtered_tokens, embeddings


def main():
    text = st.sidebar.file_uploader("Please Select to Upload a file with text", type=['txt'])
    background = open('../temp/background.txt', 'r')

    ft1, e1 = func(background)

    if text is not None:

        ft2, e2 = func(text)

        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        y1 = tsne.fit_transform(e1)
        y2 = tsne.fit_transform(e2)

        plt.scatter(y1[:, 0], y1[:, 1], c='orange')

        plt.scatter(y2[:, 0], y2[:, 1], c='blue')
        for label, x, y in zip(ft2, y2[:, 0], y2[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()
        st.pyplot()


download_model()
main()
