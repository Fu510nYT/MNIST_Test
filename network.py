import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from non_linear import *

class MyNeuralNetwork(nn.Module):
    global input_size, hidden_size, num_classes
    def __init__(self, input_size):