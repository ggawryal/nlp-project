from embedding import gensim_embedding


class Word2VecEmbedding(gensim_embedding.GensimEmbedding):
    def __init__(self):
        super().__init__("word2vec-google-news-300")
