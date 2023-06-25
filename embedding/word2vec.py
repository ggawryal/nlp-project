import logging

from embedding import WordEmbedding, UnknownWordException

import gensim.downloader


class Word2VecEmbedding(WordEmbedding):
    model = gensim.downloader.load('fasttext-wiki-news-subwords-300')

    def get_dictionary(self):
        # logging.INFO(f'Loading dictionary of {len(list(self.model.key_to_index.keys()))} words')  # todo: needs fixing
        return list(self.model.key_to_index.keys())

    def embed(self, word):
        if word not in self.model.key_to_index:
            raise UnknownWordException()
        return self.model[word]
