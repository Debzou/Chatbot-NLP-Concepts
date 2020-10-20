import json
from nltk_utils import tokenization, stemming, matrice_of_word

#open the json file
with open('data_bot.json',"r") as f:
    data_bot = json.load(f)

# define array
words = []
tags = []
matrix = []

# process the json file 
for intent in data_bot['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        # notice extend is append for a array
        words.extend(w)
        matrix.append((w,tag))

remove_punctuation = ['?','.',',',';','!']
# remove all punctuation 
words = [stemming(w) for w in words if w not in remove_punctuation]
# remove the duplicate words
words = sorted(set(words))
tags = sorted(set(tags))
print(tags)

# 10min