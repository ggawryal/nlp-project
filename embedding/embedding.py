from typing import List


class UnknownWordException(Exception):
    pass


class WordEmbedding:
    def get_dictionary(self) -> List[str]:
        raise NotImplementedError()

    def embed(self, word: str):
        raise NotImplementedError()
    
    def get_most_similar(self, word, topn):
        raise NotImplementedError()
