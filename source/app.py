import streamlit as st
from downloads import download_model
from thesaurus import Thesaurus

MODEL = 'en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0'


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Welcome to Thesaurus Visualization")
st.write(" ------ ")

download_model()

foreground = st.sidebar.file_uploader("Please select a file with the text you want to visualize", type=['txt'], key=1)
background = st.sidebar.file_uploader("Please select a file with the text for the background", type=['txt'], key=2)

if (foreground is not None) and (background is not None):

    obj = Thesaurus()

    foreground = obj.read_text(foreground)
    background = obj.read_text(background)

    obj.set_spacy_model(MODEL)

    lemmatized_f = obj.lemmatize(foreground, len(foreground))

    lemmatized_b = obj.lemmatize(background, len(background))

    tokenized_f = obj.tokenize(lemmatized_f)

    tokenized_b = obj.tokenize(lemmatized_b)

    filtered_tokens_f, filtered_tokens_f_set = obj.remove_stopwords(tokenized_f)
    filtered_tokens_b, filtered_tokens_b_set = obj.remove_stopwords(tokenized_b)

    embeddings_f = obj.make_embeddings(filtered_tokens_f_set)
    embeddings_b = obj.make_embeddings(filtered_tokens_b_set)

    fig = obj.plot_bokeh(embeddings_f, embeddings_b, filtered_tokens_f_set, filtered_tokens_b_set)

    st.bokeh_chart(fig)
