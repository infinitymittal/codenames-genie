from fasttext import FastText

board_words = set(["casino", "shot", "spurs", "blacksmith", "lawyer",\
                   "werewolf", "lion", "post", "lemonade", "sun",\
                   "hook", "pepper", "cheese", "robin", "page",\
                   "rust", "ball", "peanut", "key", "mole",\
                   "van", "sack", "piano", "santa", "screen"])
guess_words = set(["metal", "money", "animal"])

all_words = set(board_words)
all_words.update(guess_words)

def find_closest(embed_obj, guess_word):
    closest = embed_obj.find_closest_by_word(guess_word)
    closest = [word for word in closest if word[0] in board_words]
    return closest

fasttext = FastText(all_words)
for guess_word in guess_words:
    print("checking for guess {}.".format(guess_word))
    result = find_closest(fasttext, guess_word)
    for word,score in result:
        print(word, score, sep='\t')
    print()
    
