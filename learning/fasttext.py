# This file contains code to read embeddings generated via fasttext.
# Refer to https://fasttext.cc/

import os.path
import numpy as np

class FastText:
    _EMBEDDINGS_FILE="wiki-news-300d-1M.vec"
    _FASTEXT_URL = "https://fasttext.cc/docs/en/english-vectors.html"
    
    def __init__(self, all_words = None):
        if(not os.path.exists(FastText._EMBEDDINGS_FILE)):
            raise Exception("Embeddings file not found. Obtain {} from {}."\
                  .format(FastText._EMBEDDINGS_FILE, FastText._FastText_URL))
        
        self.embeddings = {}
        with open(FastText._EMBEDDINGS_FILE, encoding='utf-8', mode='r') as file:
            for line in file:
                tokens = line.split(' ')
                word = tokens[0]
                if all_words and word not in all_words:
                    continue
                embedding = np.asarray(tokens[1:], dtype='float')
                self.embeddings[word] = embedding
        print("Loaded embeddings from {}.".format(FastText._EMBEDDINGS_FILE))
    
    def get_embeddings(self):
        return self.embeddings
    
    def find_closest_by_word(self, word):
        scores = {match_word:self.cosine_dist_by_word(match_word, word)\
                  for match_word in self.embeddings.keys()}
        return sorted(scores.items(), key=lambda x:x[1], reverse=True)

    def cosine_dist_by_word(self, word1, word2):
        return FastText.cosine_dist(self.embeddings[word1], self.embeddings[word2])

    def cosine_dist(embedding1, embedding2):
        return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
