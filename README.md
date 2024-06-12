
# SkipGram Word2Vec Implementation

This project implements the SkipGram model for Word2Vec in Python, allowing users to train word embeddings, compute word similarities, and solve analogy tasks for natural language processing applications.

![SkipGram Word2Vec](https://github.com/omri898/skipgram-w2v/blob/main/DALL%C2%B7E%202024-06-12%2018.23.01%20-%20An%20artistic%20visualization%20of%20a%20neural%20network%20processing%20text%20data%20for%20word%20embeddings%20in%20natural%20language%20processing.%20The%20image%20features%20interconnect.webp)

## Overview

The SkipGram model is a neural network-based approach to learning word embeddings, where the goal is to predict the context words given a target word. This implementation allows you to train your own word embeddings using a corpus of text.

## Installation

To install the necessary dependencies, ensure you have the following files:

- [skipgram_w2v.py](https://github.com/omri898/skipgram-w2v/blob/main/skipgram_w2v.py)
- [requirements.txt](https://github.com/omri898/skipgram-w2v/blob/main/requirements.txt)

Then, run:

```sh
pip install -r requirements.txt
```

## Usage

### Normalize Text

You can normalize your text file with:

```python
from skipgram_w2v import normalize_text

sentences = normalize_text('path_to_your_text_file.txt')
```

### Train the SkipGram Model

Create an instance of the `SkipGram` class and train the model:

```python
from skipgram_w2v import SkipGram

# Normalize your text and prepare sentences
sentences = normalize_text('path_to_your_text_file.txt')

# Create the SkipGram model
model = SkipGram(sentences)

# Train the model
model.learn_embeddings()
```

### Compute Similarity

Compute the cosine similarity between two words:

```python
similarity = model.compute_similarity('word1', 'word2')
print(f"Similarity between 'word1' and 'word2': {similarity}")
```

### Find Closest Words

Find the closest words to a given word:

```python
closest_words = model.get_closest_words('word', n=5)
print(f"Closest words to 'word': {closest_words}")
```

### Analogy Test

Find a word that matches the analogy 'w1 is to w2 as w3 is to ____':

```python
analogy_word = model.find_analogy('man', 'king', 'woman')
print(f"'man' is to 'king' as 'woman' is to '{analogy_word}'")
```

## Saving and Loading the Model

### Save the Model

You can save the trained model to a file:

```python
model.learn_embeddings(model_path='skipgram_model.pkl')
```

### Load the Model

You can load a saved model from a file:

```python
from skipgram_w2v import load_model

loaded_model = load_model('skipgram_model.pkl')
```

## Requirements

- Python 3.x
- `nltk`
- `numpy`
