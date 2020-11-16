# %%
# data tool
import json
import numpy as np
import random

# file nltk_utils.py
from nltk_utils import tokenization, stemming, matrice_of_word

# library for ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# name bot
bot_name = "Treebot"

#open the json file
with open('train.json',"r") as f:
    data_bot = json.load(f)

# define array
all_words = []
tags = []
matrix = []

# process the json file 
for intent in data_bot['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        # notice extend is append for a array
        all_words.extend(w)
        matrix.append((w,tag))

remove_punctuation = ['?','.',',',';','!']
# remove all punctuation 
all_words = [stemming(w) for w in all_words if w not in remove_punctuation]
# remove the duplicate words (tips)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Binary list  (sentence by sentence)
x_train = []
# Tag position (ref tags)
y_train = []

for (pattern,tag) in matrix:
    # sentence transform into binary word 
    binList = matrice_of_word(pattern,all_words)
    # add this sentence in dim1
    x_train.append(binList)
    # find the tag corresponding to the sentence
    label = tags.index(tag)
    # add this sentence in dim2
    y_train.append(label)
    # print(binList,label)

x_train = np.array(x_train)
y_train = np.array(y_train)


# training the decision tree
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=100)
decision_tree = decision_tree.fit(x_train, y_train)


print("Let's start! ('quit' => exit)")
print(len(tags), "tags:", tags)
# use to compute the accuracy score
# the true prediction
y_true = []
# the bot prediction
y_pred = []
while True:
    # sentence = "do you use credit cards?"
    sentence = input("\nYou: ")
    if sentence == "quit":
        break
    # transform input
    sentence = tokenization(sentence)
    X = matrice_of_word(sentence, all_words)
    # prediction
    tag = decision_tree.predict([X])
    tag = tags[tag.item()]
    for intent in data_bot['intents']:
        if tag == intent["tag"]:
             print(f"{bot_name}: {random.choice(intent['responses'])}")
             print(f"{bot_name}: the tag is {tag}\n")
    # the user confirme the answere 
    y_pred.append(tag)
    correction = input("Help us to correct the bot ! \nWhat do you think is the best tag for your sentence? : ")
    y_true.append(correction)

# score 
print(f"the bot has a accuracy of {accuracy_score(y_true, y_pred)} for this section ")
