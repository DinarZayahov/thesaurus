import pickle
import nltk
import os
from tqdm import tqdm
from thesaurus import Thesaurus
from downloads import download_model
from bokeh.io import show
from transformers import logging

logging.set_verbosity_error()

download_model()

nltk.download('words')
nltk.download('wordnet')

path = '../data/'
with open(path+'2017', 'rb') as f_2017:
    data_2017 = pickle.load(f_2017)

with open(path+'2020', 'rb') as f_2020:
    data_2020 = pickle.load(f_2020)

with open(path+'2010', 'rb') as f_2010:
    data_2010 = pickle.load(f_2010)

with open(path+'shakespeare.txt', 'r') as f_shakespeare:
    shakespeare = f_shakespeare.read()

# texts is the dictionary with all source texts that will be preprocessed
# texts = {'foreground name1': [source text1, source text2, ...], ...}
texts = dict()

# list of foreground names
# foreground_names = ['foreground name1', ...]
foreground_names = []

# dictionary with the embeddings and tokens of each foreground unit
# processed_foregrounds = {'foreground_name1': {'embeds': embeddings, 'words': tokens}, ...}
processed_foregrounds = dict()

# preprocess Physics articles
foreground_name1 = 'Physics articles 2017'

texts[foreground_name1] = []

foreground_names.append(foreground_name1)

num_of_articles = 10
for physics_article in data_2017[:num_of_articles]:
    try:
        texts[foreground_name1].append(physics_article['clean'])
    except KeyError:
        continue

# preprocess Shakespeare's poem
foreground_name2 = 'Lover\'s Complaint by William Shakespeare'

texts[foreground_name2] = [shakespeare]

foreground_names.append(foreground_name2)

MODEL = 'en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0'

obj = Thesaurus()
obj.set_spacy_model(MODEL)

for foreground_unit in tqdm(foreground_names):

    all_embeddings_of_unit, all_words_of_unit = obj.process_texts(texts[foreground_unit])

    one_processed_foreground = {'embeds': all_embeddings_of_unit, 'words': all_words_of_unit}
    processed_foregrounds[foreground_unit] = one_processed_foreground

background_texts = []
if os.path.isfile(path+'coca_embeds.pickle') and os.path.isfile(path+'coca_tokens.pickle'):
    embeds = open(path+'coca_embeds.pickle', 'rb')
    background_embeds = pickle.load(embeds)

    tokens = open(path+'coca_tokens.pickle', 'rb')
    background_words = pickle.load(tokens)
else:

    background_embeds, background_words = obj.process_texts(background_texts)


fig, som = obj.plot_bokeh(background_embeds, background_words, foreground_names, processed_foregrounds)
show(fig)
