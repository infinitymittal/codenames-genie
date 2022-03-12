from fasttext import FastText

board_words = set(["casino", "shot", "spurs", "blacksmith", "lawyer",\
                   "werewolf", "lion", "post", "lemonade", "sun",\
                   "hook", "pepper", "cheese", "robin", "page",\
                   "rust", "ball", "peanut", "key", "mole",\
                   "van", "sack", "piano", "santa", "screen"])
guess_word = "metal"

all_words = set(board_words)
all_words.add(guess_word)

def find_closest(embed_obj, word):
    closest = embed_obj.find_closest_by_word(word)
    #closest = [word for word in closest if word[0] in board_words]
    return closest

fasttext = FastText(all_words)
result = find_closest(fasttext, guess_word)
for word,score in result:
    print(word, score, sep='\t')

