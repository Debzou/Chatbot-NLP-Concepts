# %%
# data tool
import json
import numpy as np

# file nltk_utils.py
from nltk_utils import tokenization, stemming, matrice_of_word

# library for ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text



#open the json file
with open('data_bot.json',"r") as f:
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

print(len(matrix), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

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
decision_tree = DecisionTreeClassifier(random_state=1, max_depth=57)
decision_tree = decision_tree.fit(x_train, y_train)

# display graph
r = export_text(decision_tree)
print(r)

while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    # transform input
    sentence = tokenization(sentence)
    X = matrice_of_word(sentence, all_words)
    print(np.array(X))
    print(x_train[0])
    # prediction
    tag = decision_tree.predict([X])
    print(tag)
    break
# %%
