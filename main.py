import random
from typing import List

from similarity import WordSimilarityScorer, CosineSimilarityScorer
from embedding import UnknownWordException, word2vec


class Game:
    UNKNOWN_WORD = -1
    GUESSED = -2

    def __init__(self, dictionary: List[str], scorer: WordSimilarityScorer):
        self.scorer = scorer
        self.secret_word = random.choice(dictionary)
        self.ended = False
        self.guesses = []

    def turn(self, guess):
        self.guesses.append(guess)
        if guess == self.secret_word:
            self.ended = True
            return Game.GUESSED

        try:
            return self.scorer.score(self.secret_word, guess)
        except UnknownWordException:
            return Game.UNKNOWN_WORD

    def has_not_ended(self):
        return not self.ended

    def turns_passed(self):
        return len(self.guesses)


def play():
    embedding = word2vec.Word2VecEmbedding()
    scorer = CosineSimilarityScorer(embedding)

    print("Welcome to the word guessing game!")
    print("Randomly choosing a secret word...")

    game = Game(embedding.get_dictionary(), scorer)

    while game.has_not_ended():
        print("Guess the secret word!")
        guess = input("> ")
        result = game.turn(guess)
        if result == Game.UNKNOWN_WORD:
            print(f"I don't know word `{guess}`. Try again!")
        elif result == Game.GUESSED:
            print(f"You guessed the secret word in {game.turns_passed()} turns!")
            return
        else:
            print(f"Your word is {result*100:.2f}% similar to the secret word. Try again!")


if __name__ == "__main__":
    play()
