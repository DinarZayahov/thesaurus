import base64
import streamlit as st
from downloads import download_model
from thesaurus import Thesaurus
from io import StringIO, BytesIO


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

    filtered_tokens_f, filtered_tokens_f_set = obj.remove_stopwords(tokenized_f)
    filtered_tokens_b, filtered_tokens_b_set = obj.remove_stopwords(tokenized_b)

    embeddings_f = obj.get_embeddings(filtered_tokens_f_set)
    embeddings_b = obj.get_embeddings(filtered_tokens_b_set)

    y_f = obj.apply_tsne(embeddings_f)
    y_b = obj.apply_tsne(embeddings_b)

    # fig = obj.plot_pyplot(y_f, y_b, filtered_tokens_b_set)

    df1 = obj.make_dataframe(filtered_tokens_f, filtered_tokens_f_set, y_f, 'foreground')
    df2 = obj.make_dataframe(filtered_tokens_b, filtered_tokens_b_set, y_b, 'background')
    df = obj.join_dataframes(df1, df2)
    fig = obj.plot_plotly(df)

    # fig = obj.plot_plotly(df1)
    # fig2 = obj.plot_plotly(df2)

    if fig is not None:
        st.plotly_chart(fig)
        # st.plotly_chart(fig2)

        # downloading
        mybuff = StringIO()
        fig.write_html(mybuff, include_plotlyjs='cdn')
        mybuff = BytesIO(mybuff.getvalue().encode())
        b64 = base64.b64encode(mybuff.read()).decode()
        href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download plot</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.pyplot()
