# Word Guessing Game

## Description
This is a simple word guessing game. 
A word is chosen at random and the player has to guess it.

## Installation

Use pipenv sync to install all packages specified in Pipfile.lock.
```bash
pipenv sync
```

Then run the main.py file.

Good luck!


## Research log and further improvement ideas
* Test embeddings from some transformer, e.g. `Bert`,
* Fine-tune model on some dataset with more specific language, e.g. some fantasy books, and sample secret words only from that dataset,
* Develop a guesser agent using some other embedding and use the game as a measure of embedding similarity,
* Create a graphical UI
