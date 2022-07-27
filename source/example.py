from downloads import make_downloads
from thesaurus import Thesaurus

obj = Thesaurus(lang='eng')

text1 = obj.read_pickle('../data/2017')
text2 = obj.read_txt('../data/shakespeare.txt')

# texts is the dictionary with all source texts that will be preprocessed
# texts = {'foreground name1': [source text1, source text2, ...], ...}
texts = dict()

# list of foreground names
# foreground_names = ['foreground name1', ...]
foreground_names = []

foreground_name1 = 'Physics articles 2017'
texts[foreground_name1] = obj.custom_preprocessing_of_data(text1)
foreground_names.append(foreground_name1)

foreground_name2 = 'Lover\'s Complaint by William Shakespeare'
texts[foreground_name2] = [text2]
foreground_names.append(foreground_name2)

MODEL = 'en_core_web_md-3.0.0/en_core_web_md/en_core_web_md-3.0.0'
obj.set_spacy_model(MODEL)

# dictionary with the embeddings and tokens of each foreground unit
# processed_foregrounds = {'foreground_name1': {'embeds': embeddings, 'words': tokens}, ...}
processed_foregrounds = obj.process_foreground(foreground_names, texts)

background_embeds, background_words = obj.import_background()

obj.set_som()

obj.show_map(background_embeds, background_words, foreground_names, processed_foregrounds)
