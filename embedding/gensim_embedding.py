import logging
from embedding import WordEmbedding, UnknownWordException
import gensim.downloader
from pathlib import Path
import os
import pickle
from gensim.models import KeyedVectors

class GensimEmbedding(WordEmbedding):
    def __init__(self, model_name):
        model_path = os.path.join('data', model_name + '.model')
        if not Path(model_path).exists():
            logging.info(f'{model_name} model not found in {model_path} directory, dowloading it')
            self.model = gensim.downloader.load(model_name)
            logging.info('Cleaning up dataset')
            self.model_data_cleanup()
            with open(model_path, "w+b") as file:
                pickle.dump(self.model, file)
        else:
            logging.debug(f'Loading {model_name} model from {model_path}')
            with open(model_path, "rb") as file:
                self.model = pickle.load(file)
        logging.info(f'Loaded {model_name} model with {len(list(self.model.key_to_index.keys()))} words')

    def get_dictionary(self):
        return list(self.model.key_to_index.keys())

    def embed(self, word):
        if word not in self.model.key_to_index:
            raise UnknownWordException()
        return self.model[word]
    
    def model_data_cleanup(self):
        top_words = []
        vectors = []
        for (word, v) in zip(self.get_dictionary(), self.model.vectors):
            if word == word.lower() and word.isalpha() and len(word) >= 3:
                top_words.append(word)
                vectors.append(v)
            if len(top_words) >= 40000:
                break
                
        kv = KeyedVectors(vector_size = self.model.vector_size)
        kv.add_vectors(top_words, vectors)
        self.model = kv

    def get_most_similar(self, word, topn):
        return self.model.most_similar(word, topn=topn)