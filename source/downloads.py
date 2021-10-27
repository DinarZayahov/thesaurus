import requests
import os
import tarfile


def download_model():

    filename = 'en_core_web_md'
    if not os.path.exists(filename):
        r = requests.get('https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0'
                         '/en_core_web_md-3.0.0.tar.gz', allow_redirects=True)
        open(filename, 'wb').write(r.content)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()
    else:
        print(f"Model found in cache")

    return filename
