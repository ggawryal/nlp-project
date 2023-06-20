from embedding import WordEmbedding, UnknownWordException

import gensim.downloader


class Word2VecEmbedding(WordEmbedding):
    model = gensim.downloader.load('word2vec-google-news-300')

    def get_dictionary(self):
        return list(self.model.key_to_index.keys())

    def embed(self, word):
        if word not in self.model.key_to_index:
            raise UnknownWordException()
        return self.model[word]
