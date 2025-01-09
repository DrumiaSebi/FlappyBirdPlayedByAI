import torch
from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()

        # input layer is by default
        # first hidden layer, Linear specifies it is fully connected:
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # second hidden layer:
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # output:
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x), 0.01)
        x2 = F.leaky_relu(self.fc2(x1), 0.01)
        # we don't apply activation function to the output
        return self.fc3(x2)

# Test
# input_dim = 12
# output_dim = 2
# hidden_dim = 1500
# net = NeuralNetwork(input_dim, output_dim, hidden_dim)
# state = torch.rand(10, input_dim)
# output = net(state)
# print(output)