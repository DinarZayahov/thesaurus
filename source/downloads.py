import requests
import os
import tarfile
import nltk


def download_model():

    filename = 'en_core_web_md_temporary'
    if not os.path.exists(filename):
        r = requests.get('https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0'
                         '/en_core_web_md-3.0.0.tar.gz', allow_redirects=True)
        open(filename, 'wb').write(r.content)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()


def make_downloads():
    download_model()
    nltk.download('words')
    nltk.download('wordnet')

