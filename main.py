import random
from typing import List
import logging
from similarity import WordSimilarityScorer, GensimNativeSimilarityScorer
from embedding import UnknownWordException, word2vec, fasttext
import nltk

logging.basicConfig(filename='nlp-app.log', format='[%(asctime)s] %(levelname)s:%(message)s', level=logging.DEBUG)
nltk.download('averaged_perceptron_tagger', quiet=True)


class Game:
    UNKNOWN_WORD = -1
    GUESSED = -2

    def __init__(self, dictionary: List[str], scorer: WordSimilarityScorer):
        self.scorer = scorer
        self.secret_word = random.choice(dictionary)

        self.max_score_possible = scorer.score(self.secret_word, self.secret_word)
        self.top_1k_words = { word: idx+1 for idx,(word, _) in enumerate(scorer.embedding.get_most_similar(self.secret_word, 1000)) }

        self.ended = False
        self.guesses = []

    def turn(self, guess):
        self.guesses.append(guess)
        if guess == self.secret_word:
            self.ended = True
            return Game.GUESSED

        try:
            return self.scorer.score(self.secret_word, guess) / self.max_score_possible
        except UnknownWordException:
            return Game.UNKNOWN_WORD

    def get_rank_in_top_1k_words(self, word):
        if word not in self.top_1k_words:
            return None
        return self.top_1k_words[word]

    def has_not_ended(self):
        return not self.ended

    def turns_passed(self):
        return len(self.guesses)


def is_noun(word):
    if word != word.lower() or not word.isalpha() or len(word) < 3:
        return False
    tagged_word = nltk.pos_tag([word])
    return tagged_word[0][1] in {'NN', 'NNP'}


def create_secret_word_dictionary(dictionary):
    res = []
    for word in dictionary:
        if is_noun(word):
            res.append(word)
        if len(res) >= 2000:
            break

    logging.info(f'Secret word dictionary created with {len(res)} words')
    return res


def ordinal_suffix(n):
    if n%100 >= 10 and n%100 <= 20:
        return 'th'
    if n % 10 == 1:
        return 'st'
    if n % 10 == 2:
        return 'nd'
    if n % 10 == 3:
        return 'rd'
    return 'th'


def play():
    # embedding = word2vec.Word2VecEmbedding()
    embedding = fasttext.FasttextEmbedding()

    secret_word_dictionary = create_secret_word_dictionary(embedding.get_dictionary())
    scorer = GensimNativeSimilarityScorer(embedding)

    print("Welcome to the word guessing game!")
    print("Randomly choosing a secret word...")

    game = Game(secret_word_dictionary, scorer)
    best_so_far = (-10**9, -10**9, "")
    while game.has_not_ended():
        print("Guess the secret word!")
        guess = input("> ")
        if guess == "/hint":
            print(f"The secret word starts with {game.secret_word[0]}")
            continue
        if guess == "/cheat":
            print(f"The secret word is {game.secret_word}")
            continue
        if guess == "/exit":
            return
        
        result = game.turn(guess)
        if result == Game.UNKNOWN_WORD:
            print(f"I don't know word `{guess}`. Try again!")
        elif result == Game.GUESSED:
            print(f"You guessed the secret word in {game.turns_passed()} turns!")
            return
        else:
            rank = game.get_rank_in_top_1k_words(guess)
            rank_info = ""
            if rank is not None:
                rank_info = f" ({rank}{ordinal_suffix(rank)} most similar)" if rank is not None else ""

            best_so_far = max(best_so_far, (-10**9 if rank is None else -rank, result, guess))
            print(f"Your word is {result*100:.2f}% similar to the secret word{rank_info}. Best so far: {best_so_far[2]}. Try again!")


if __name__ == "__main__":
    play()
