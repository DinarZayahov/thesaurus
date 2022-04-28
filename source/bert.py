import torch
import numpy as np
from transformers import BertTokenizer, BertModel

path = '../data/'
STOPWORDS_FILE = path+'extended_stopwords.txt'


class Bert:
    def __init__(self):
        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True)
        model.eval()
        self.model = model

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer

    @staticmethod
    def bert_text_preparation(text, tokenizer):
        """Preparing the input for BERT

        Takes a string argument and performs
        pre-processing like adding special tokens,
        tokenization, tokens to ids, and tokens to
        segment ids. All tokens are mapped to segment id = 1.

        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object
                to convert text into BERT-readable tokens and ids

        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids


        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    @staticmethod
    def get_bert_embeddings(tokens_tensor, segments_tensors, model):
        """Get embeddings from an embedding model

        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens]
                with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens]
                with segment ids for each token in text
            model (obj): Embedding model to generate embeddings
                from token and segment ids

        Returns:
            list: List of list of floats of size
                [n_tokens, n_embedding_dimensions]
                containing embeddings for each token

        """

        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings = token_embeddings.permute(1, 0, 2)

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        # Converting torch tensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_vecs_sum]

        return list_token_embeddings

    @staticmethod
    def merge_vectors(list_of_vectors):
        return np.mean(list_of_vectors, axis=0)

    @staticmethod
    def merge_subwords(list_of_subwords):
        res = list_of_subwords[0]
        for subword in list_of_subwords[1:]:
            res += subword[2:]
        return res

    @staticmethod
    def get_stopwords(path_):
        stopwords_file = open(path_, 'r')
        stopwords = []
        for line in stopwords_file:
            stopwords.append(line[:-1])
        return stopwords

    def f(self, sentences):
        all_tokenized_sentences = []
        all_embeddings = []

        # for sentence in tqdm(sentences):
        for sentence in sentences:
            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(sentence, self.tokenizer)
            list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors, self.model)

            all_tokenized_sentences += tokenized_text
            all_embeddings += list_token_embeddings

        return all_tokenized_sentences, all_embeddings

    def merge(self, tokens, embeddings):
        ranges = []
        i = -1
        while i >= -len(tokens):
            if tokens[i][:2] == "##":
                start = i
                while tokens[i][:2] == "##":
                    i -= 1
                end = i
                ranges.insert(0, (start, end))
            else:
                i -= 1

        for r in ranges:
            tokens = tokens[-len(tokens):r[1]] + [self.merge_subwords(tokens[r[1]:r[0] + 1])] + tokens[r[0] + 1:]
            embeddings = embeddings[-len(embeddings):r[1]] + [self.merge_vectors(embeddings[r[1]:r[0] + 1])] + \
                embeddings[r[0] + 1:]

        return tokens, embeddings

    def filter(self, tokens, embeddings):
        stopwords = self.get_stopwords(STOPWORDS_FILE)

        filtered_tokens = []
        filtered_embeddings = []
        for i in range(len(tokens)):
            if tokens[i] not in ['[SEP]', '[CLS]'] and tokens[i].isalpha() and tokens[i] not in stopwords:
                filtered_tokens.append(tokens[i])
                filtered_embeddings.append(embeddings[i])

        return filtered_tokens, filtered_embeddings

    def bert_embedding(self, token):
        a, b = self.f([token])
        a, b = self.merge(a, b)
        a, b = self.filter(a, b)
        emb = b[0]
        return emb[:300]
