import json
import numpy as np
from nltk_utils import tokenization, stemming, matrice_of_word

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


# Binary list  (sentence by sentence)
dim1_train = []
# Tag position (ref tags)
dim2_train = []

for (pattern,tag) in matrix:
    binList = matrice_of_word(pattern,words)
    dim1_train.append(binList)

    label = tags.index(tag)
    dim2_train.append(label)

dim1_train = np.array(dim1_train)
dim2_train = np.array(dim2_train)

print(dim1_train)
print(dim2_train)

#20min