from embedding import gensim_embedding


class FasttextEmbedding(gensim_embedding.GensimEmbedding):
    def __init__(self):
        super().__init__("fasttext-wiki-news-subwords-300")
