import logging
from embedding import WordEmbedding, UnknownWordException
import gensim.downloader
from pathlib import Path
import os
import pickle


class GensimEmbedding(WordEmbedding):
    def __init__(self, model_name):
        model_path = os.path.join('data', model_name + '.model')
        if not Path(model_path).exists():
            logging.info(f'{model_name} model not found in {model_path} directory, dowloading it')
            self.model = gensim.downloader.load(model_name)
            with open(model_path, "w+b") as file:
                pickle.dump(self.model, file)
        else:
            with open(model_path, "rb") as file:
                self.model = pickle.load(file)
            logging.debug(f'Loading {model_name} model from {model_path}')
        logging.info(f'Loaded {model_name} model with {len(list(self.model.key_to_index.keys()))} words')

    def get_dictionary(self):
        return list(self.model.key_to_index.keys())

    def embed(self, word):
        if word not in self.model.key_to_index:
            raise UnknownWordException()
        return self.model[word]
