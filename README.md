# Thesaurus Visualization

## Current available notebooks
``ThesaurusVisualization.ipynb`` Is a demo on how to use the library (English language)

``Tutorial.ipynb`` Shows how to train the model

``MultiLang.ipynb`` Multi languages demo (French, German, Russian and Arabic)

``back.ipynb`` Shows how to prepare and process the data for training
## How to install and run
Install the requirements:

```
pip install -r requirements.txt
```

cd into source directory:

```
%cd source
```

Import the library:

```
from thesaurus import Thesaurus
```

Create an object and specify the language:

```
obj = Thesaurus(lang='eng')
```

Current supported languages are:

- English ``eng``
- French ``fra``
- German ``deu``
- Arabic ``ara``
- Russian ``ru``

Chose one or more files to process as a foreground:

```
text1 = obj.read_txt('../data/texts/covid19.txt')
text2 = obj.read_txt('../data/texts/descartes.txt') 
```
```
texts = dict()

foreground_names = []

foreground_name1 = 'Covid 19'
texts[foreground_name1] = [text1]
foreground_names.append(foreground_name1)

foreground_name2 = 'Discours de la méthode, René Descartes'
texts[foreground_name2] = [text2]
foreground_names.append(foreground_name2)
```
```
processed_foregrounds = obj.process_foreground(foreground_names, texts)
```
Import the background files:
```
background_embeds, background_words = obj.import_background()
```
Note: If embeddings aren't available for the background then they will be auto generated

Visualize the texts:
```
print("Go to https://nbviewer.org/github/DinarZayahov/thesaurus/blob/master/ThesaurusVisualization.ipynb 
       to see the outputs of the notebook as the GitHub doesn't render dynamic output")
obj.show_map(background_embeds, background_words, foreground_names, processed_foregrounds)
```
## Download embeddings on Colab:

After you change directory to '/source' in google colab run the current cell,
this cell downloads the files into their folders:

```
!gdown 1pJnTdw5qyitFNVC_FLfmjhNdEBk0fH0N -O ../data/back_embeds/
!gdown 1BLHIb82kTFuSzkyEbe1f044UGojXuhV8 -O ../data/back_embeds/
!gdown 1WrU9i5BiCckDebYAcOZu83ZL1UD9kvM2 -O ../data/back_embeds/
```