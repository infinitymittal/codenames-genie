from glove import Glove

board_words = set(["casino", "shot", "spurs", "blacksmith", "lawyer",\
                   "werewolf", "lion", "post", "lemonade", "sun",\
                   "hook", "pepper", "cheese", "robin", "page",\
                   "rust", "ball", "peanut", "key", "mole",\
                   "van", "sack", "piano", "santa", "screen"])
guess_word = "metal"

all_words = set(board_words)
all_words.add(guess_word)

glove = Glove(all_words)

def find_closest(word):
    closest = glove.find_closest_by_word(word)
    #closest = [word for word in closest if word[0] in board_words]
    return closest

# result = find_closest(guess_word)
# for word,score in result:
#     print(word, score, sep='\t')

print(glove.get_embeddings()["metal"])
print(glove.get_embeddings()["blacksmith"])
print(glove.get_embeddings()["metal"]*glove.get_embeddings()["blacksmith"])