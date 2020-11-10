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
# remove the duplicate words (tips)
words = sorted(set(words))
tags = sorted(set(tags))


# Binary list  (sentence by sentence)
dim1_train = []
# Tag position (ref tags)
dim2_train = []

for (pattern,tag) in matrix:
    # sentence transform into binary word 
    binList = matrice_of_word(pattern,words)
    # add this sentence in dim1
    dim1_train.append(binList)
    # find the tag corresponding to the sentence
    label = tags.index(tag)
    # add this sentence in dim2
    dim2_train.append(label)
    # print(binList,label)

dim1_train = np.array(dim1_train)
dim2_train = np.array(dim2_train)

print(dim1_train)
print(dim2_train)


class DataOfChatbot(Dataset):
    """
    dataset is standarding by a model
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    def __init__(self):
        self.n = len(dim1_train)
        self.dim1_train = dim1_train
        self.dim2_train = dim2_train
    # gather element 
    def __getitem__(self, index):
        return self.dim1_train[index],self.dim2_train[index]
    # size 
    def __len__(self):
        return self.n    

# hyperparameters
batch_size = 8 # split the dataset in 8

dataset = DataOfChatbot()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2) # 2 threads
