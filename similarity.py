
from embedding import WordEmbedding, UnknownWordException

class WordSimilarityScorer:
    def __init__(self, embedding: WordEmbedding):
        self.embedding = embedding


    def score(self, first: str, second: str) -> float:
        """
        Returns a score between 0 and 1, where 1 means the words are identical.
        Should be symmetric, i.e. score(first, second) == score(second, first).
        Raises UnknownWordException if either word is not in the scorer's dictionary.
        """
        raise NotImplementedError()
    

class CosineSimilarityScorer(WordSimilarityScorer):
    def score(self, first: str, second: str) -> float:
        for word in [first, second]:
            if word not in self.embedding.get_dictionary():
                raise UnknownWordException(f"Unknown word: {word}")

        return sum(a*b for a,b in zip(self.embedding.embed(first), self.embedding.embed(second)))
    