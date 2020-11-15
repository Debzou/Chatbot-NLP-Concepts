# data tool
import json
import numpy as np

# file nltk_utils.py
from nltk_utils import tokenization, stemming, matrice_of_word
# file modeling.py
from modeling import NeuralNetworkBOT

# library for ML
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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

# print(x_train)
# print(y_train)


class DataOfChatbot(Dataset):
    """
    dataset is standarding by a model
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    def __init__(self):
        self.n = len(x_train)
        # input (binary sentence)
        self.x_train = x_train
        # output (tag)
        self.y_train = y_train
    # gather element 
    def __getitem__(self, index):
        return torch.tensor(self.x_train[index]).float(),self.y_train[index]
    # size 
    def __len__(self):
        return self.n    


# hyperparameters (neuralnet parm)
batch_size = 8 # split the dataset in 8
input_size = len(x_train[0]) # all bag of word as the same size 
hidden_size = 8
number_class = len(tags)
learning_rate = 0.001
number_epochs = 1000
# other value
loss = 1
FILE = "data.pth" #  file for the model


# create dataset
dataset = DataOfChatbot()
# load the dataset
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True) 

# cuda (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create the model
model = NeuralNetworkBOT(input_size,hidden_size,number_class).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # entropy loss (log loss) [1 is bad prediction and 0 is a good prediction]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimization

# Train the neural network model
for epoch in range(number_epochs):
    for (trainwords, trainlabels) in train_loader:
        trainwords = trainwords.to(device)
        trainlabels = trainlabels.to(dtype=torch.long).to(device)
        output = model(trainwords)    # pass the data forward
        loss = criterion(output, trainlabels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{number_epochs}], Loss: {loss.item()}')

print(f'final loss: {loss.item()}')

# save the model
# create data
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"number_class": number_class,
"all_words": all_words,
"tags": tags
}

torch.save(data, FILE)

print(f'the training is finish ! the file will save in {FILE}')