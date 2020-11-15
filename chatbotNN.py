import random
import json
import torch

from modeling import NeuralNetworkBOT
from nltk_utils import tokenization, matrice_of_word

# in order to use cuda (GPU processing)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open the chatbot file
with open('data_bot.json', 'r') as jsonfile:
    data_bot = json.load(jsonfile)

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

# bot's name
bot_name = "DebzouBot"

print("Let's start! ('quit' => exit)")

while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
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
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")



