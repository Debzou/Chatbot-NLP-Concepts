# data tool
import random
import json
import torch

# library for ML
from modelingNN import NeuralNetworkBOT
# file nltk_utils.py
from nltk_utils import tokenization, matrice_of_word
# compute the accuracy score
from sklearn.metrics import accuracy_score

# in order to use cuda (GPU processing)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open the chatbot file
with open('train.json', 'r') as jsonfile:
    data_bot = json.load(jsonfile)

# bot's name
bot_name = "NNBot"

# open the trainning model
FILE = "data.pth"
data = torch.load(FILE)

# extract information 
input_size = data["input_size"]
hidden_size = data["hidden_size"]
number_class= data["number_class"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# load the model
model = NeuralNetworkBOT(input_size, hidden_size,number_class).to(device)
model.load_state_dict(model_state)
model.eval()

# use to compute the accuracy score
# the true prediction
y_true = []
# the bot prediction
y_pred = []
print("Let's start! ('quit' => exit)")
print(len(tags), "tags:", tags)
# chat with the bot
while True:
    # sentence = "do you use credit cards?"
    sentence = input("\nYou: ")
    if sentence == "quit":
        break
    # transform input
    sentence = tokenization(sentence)
    X = matrice_of_word(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)
    output = model(torch.tensor(X))
    # prediction
    _, predicted = torch.max(output, dim=1)
    # gather tag 
    tag = tags[predicted.item()]
    # choose a randon answere 
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in data_bot['intents']:
            if tag == intent["tag"]:
                print(f"\n{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
        # the user confirme the answere 
    y_pred.append(tag)
    correction = input("\nHelp us to correct the bot ! \nWhat do you think is the best tag for your sentence? : ")
    y_true.append(correction)

# score 
print(f"the bot has a accuracy of {accuracy_score(y_true, y_pred)} for this section ")



