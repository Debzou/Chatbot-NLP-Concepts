import torch
import torch.nn as nn

class NeuralNetworkBOT(nn.Module):
    def __init__(self,input_size,hidden_size,number_class):
        super(NeuralNetworkBOT,self).__init__()
        # input layers
        self.l1 = nn.Linear(input_size,hidden_size)
        # hidden layers
        self.l2 = nn.Linear(hidden_size,hidden_size)
        # output layers
        self.l3 = nn.Linear(hidden_size,number_class)
        # activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

      