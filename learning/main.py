from glove import Glove

board_words = set(["casino", "shot", "spurs", "blacksmith", "lawyer",\
                   "werewolf", "lion", "post", "lemonade", "sun",\
                   "hook", "pepper", "cheese", "robin", "page",\
                   "rust", "ball", "peanut", "key", "mole",\
                   "van", "sack", "piano", "santa", "screen"])

print(len(board_words))

glove = Glove()

def find_closest(word):
    closest = glove.find_closest_by_word(word)
    closest = [word for word in closest if word[0] in board_words]
    return closest

print(find_closest("metal"))
