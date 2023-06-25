import random
from typing import List
import logging
from similarity import WordSimilarityScorer, CosineSimilarityScorer
from embedding import UnknownWordException, word2vec
import nltk

nltk.download('averaged_perceptron_tagger')
logging.basicConfig(filename='nlp-app.log', format='[%(asctime)s] %(levelname)s:%(message)s', level=logging.DEBUG)


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


def is_noun(word):
    if word != word.lower() or not word.isalpha():
        return False
    tagged_word = nltk.pos_tag([word])
    return tagged_word[0][1] in {'NN', 'NNP'}


def create_secret_word_dictionary(dictionary):
    res = []
    for word in dictionary:
        if is_noun(word):
            res.append(word)
        if len(res) >= 5000:
            break

    logging.info(f'Secret word dictionary created with {len(res)} words')
    return res


def play():
    embedding = word2vec.Word2VecEmbedding()
    secret_word_dictionary = create_secret_word_dictionary(embedding.get_dictionary())
    scorer = CosineSimilarityScorer(embedding)
    print(secret_word_dictionary)
    print("Welcome to the word guessing game!")
    print("Randomly choosing a secret word...")

    game = Game(secret_word_dictionary, scorer)

    while game.has_not_ended():
        print("Guess the secret word!")
        guess = input("> ")
        if guess == "/hint":
            print(f"The secret word starts with {game.secret_word[0]}")
            continue
        if guess == "/cheat":
            print(f"The secret word is {game.secret_word}")
            continue
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
