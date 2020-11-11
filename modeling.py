import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,number_class):
        super(NeuralNetwork,self).__init__()
        # input layers
        self.l1 = nn.Linear(input_size,hidden_size)
        # hidden layers
        self.l2 = nn.Linear(hidden_size,hidden_size)
        # output layers
        self.l3 = nn.Linear(hidden_size,number_class)
        # activation function
        self.relu = nn.ReLU()
    
    def forward(self,x):
        # Pass the input tensor through each of our operations
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        return out

      