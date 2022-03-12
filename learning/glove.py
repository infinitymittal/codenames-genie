# This file contains code to read embeddings generated via glove.
# Refer to https://nlp.stanford.edu/projects/glove/

import os.path
import numpy as np

class Glove:
    _EMBEDDINGS_FILE="glove.6B.50d.txt"
    _GLOVE_URL = "https://nlp.stanford.edu/projects/glove"
    
    def __init__(self):
        if(not os.path.exists(Glove._EMBEDDINGS_FILE)):
            raise Exception("Embeddings file not found. Obtain {} from {}."\
                  .format(Glove._EMBEDDINGS_FILE, Glove._GLOVE_URL))
        
        self.embeddings = {}
        with open(Glove._EMBEDDINGS_FILE, encoding='utf-8', mode='r') as file:
            for line in file:
                tokens = line.split(' ')
                embedding = np.asarray(tokens[1:], dtype='float')
                self.embeddings[tokens[0]] = embedding
    
    def get_embeddings(self):
        return self.embeddings
    
    def find_closest_by_word(self, word):
        return sorted(self.embeddings.keys(), key=lambda match_word: Glove.cosine_dist(self.embeddings[match_word], self.embeddings[word]), reverse=True)

    def cosine_dist(embedding1, embedding2):
        return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
