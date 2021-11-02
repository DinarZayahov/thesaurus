import streamlit as st
from downloads import download_model
from thesaurus import Thesaurus


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

    obj.set_spacy_model('en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0')

    lemmatized_f = obj.lemmatize(foreground)
    lemmatized_b = obj.lemmatize(background)

    tokenized_f = obj.tokenize(lemmatized_f)
    tokenized_b = obj.tokenize(lemmatized_b)

    filtered_tokens_f = obj.remove_stopwords(tokenized_f)
    filtered_tokens_b = obj.remove_stopwords(tokenized_b)

    embeddings_f = obj.get_embeddings(filtered_tokens_f)
    embeddings_b = obj.get_embeddings(filtered_tokens_b)

    y_f = obj.apply_tsne(embeddings_f)
    y_b = obj.apply_tsne(embeddings_b)

    obj.plot(y_f, y_b, filtered_tokens_b)

    st.pyplot()
