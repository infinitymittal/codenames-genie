# This file contains code to read embeddings generated via glove.
# Refer to https://nlp.stanford.edu/projects/glove/

import os.path
import numpy as np

class Glove:
    _EMBEDDINGS_FILE="glove.6B.300d.txt"
    _GLOVE_URL = "https://nlp.stanford.edu/projects/glove"
    
    def __init__(self, all_words = None):
        if(not os.path.exists(Glove._EMBEDDINGS_FILE)):
            raise Exception("Embeddings file not found. Obtain {} from {}."\
                  .format(Glove._EMBEDDINGS_FILE, Glove._GLOVE_URL))
        
        self.embeddings = {}
        with open(Glove._EMBEDDINGS_FILE, encoding='utf-8', mode='r') as file:
            for line in file:
                tokens = line.split(' ')
                word = tokens[0]
                if all_words and word not in all_words:
                    continue
                embedding = np.asarray(tokens[1:], dtype='float')
                self.embeddings[word] = embedding
    
    
    
    def get_embeddings(self):
        return self.embeddings
    
    def find_closest_by_word(self, word):
        scores = {match_word:self.cosine_dist_by_word(match_word, word)\
                  for match_word in self.embeddings.keys()}
        return sorted(scores.items(), key=lambda x:x[1], reverse=True)

    def cosine_dist_by_word(self, word1, word2):
        return Glove.cosine_dist(self.embeddings[word1], self.embeddings[word2])

    def cosine_dist(embedding1, embedding2):
        return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
