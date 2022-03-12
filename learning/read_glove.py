# This file contains code to read embeddings generated via glove.
# Refer to https://nlp.stanford.edu/projects/glove/

import os.path
import sys
import numpy as np

_EMBEDDINGS_FILE="glove.6B.50d.txt"
_GLOVE_URL = "https://nlp.stanford.edu/projects/glove"

if(not os.path.exists(_EMBEDDINGS_FILE)):
    print("Embeddings file not found. Obtain {} from {}.".format(_EMBEDDINGS_FILE, _GLOVE_URL))
    sys.exit(1)

embeddings = {}
with open(_EMBEDDINGS_FILE, encoding='utf-8', mode='r') as file:
    i=0
    for line in file:
        if i==0:
            tokens = line.split(' ')
            embedding = np.asarray(tokens[1:], dtype='float')
            embeddings[tokens[0]] = embedding

def find_closest_by_word(main_word):
    return sorted(embeddings.keys(), key=lambda word: cosine_dist(embeddings[word], embeddings[main_word]), reverse=True)

def cosine_dist(embedding1, embedding2):
    return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
        
print(embeddings['man'])
print(find_closest_by_word("man")[:10])
print(find_closest_by_word("woman")[:10])


