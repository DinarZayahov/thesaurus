import requests
import os
import tarfile


def download_model_en():
    filename = 'en_core_web_md_temporary'
    if not os.path.exists(filename):
        r = requests.get('https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0'
                         '/en_core_web_md-3.0.0.tar.gz', allow_redirects=True)
        open(filename, 'wb').write(r.content)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()


def download_model_fr():
    filename = 'fr_core_news_md_temporary'
    if not os.path.exists(filename):
        r = requests.get('https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.3.0'
                         '/fr_core_news_md-3.3.0.tar.gz', allow_redirects=True)
        open(filename, 'wb').write(r.content)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()


def download_model_ru():
    filename = 'ru_core_news_md_temporary'
    if not os.path.exists(filename):
        r = requests.get('https://github.com/explosion/spacy-models/releases/download/ru_core_news_md-3.4.0'
                         '/ru_core_news_md-3.4.0.tar.gz', allow_redirects=True)
        open(filename, 'wb').write(r.content)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()


def make_downloads(lang):
    if lang == 'eng':
        download_model_en()
    elif lang == 'fra':
        download_model_fr()
    elif lang == 'rus':
        download_model_ru()
    # else:
    #     print("Please choose one of the following languages: ['eng','fra']")
