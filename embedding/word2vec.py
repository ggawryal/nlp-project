from embedding import WordEmbedding

class Word2VecEmbedding(WordEmbedding):
    def get_dictionary(self):
        raise NotImplementedError()

    def embed(self, word):
        raise NotImplementedError()