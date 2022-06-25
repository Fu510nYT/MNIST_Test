import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Courtesy to https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/

# Creating the Network (One Hidden Layer)
class MyNeuralNetwork(nn.Module):
    def __init__(self, input_layer_size, hidden_size, num_classes):
        super(MyNeuralNetwork, self).__init__()
        self.input_size = input_layer_size
        self.l1 = nn.Linear(input_layer_size,
                            hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,
                            num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Init Network Parameters
input_layer_size = 784
hidden_layers_size = 500
classes_num = 10
epochs = 2
batch_size = 100
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

# Calling the MNIST Dataset from PyTorch

training_mnist = torchvision.datasets.MNIST(root="./data",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
testing_mnist = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor())
# Specifying Batch Size and Importing Data Loader
train_loader = torch.utils.data.DataLoader(dataset=training_mnist,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testing_mnist,
                                          batch_size=batch_size,
                                          shuffle=False)
# Creating the Model
model = MyNeuralNetwork(input_layer_size, hidden_layers_size, classes_num).to(device)

# Init Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
total_loops = len(train_loader)
for epoch in range(epochs):
    for i, (image, labels) in enumerate(train_loader):
        # Reshaping Images
        image = image.reshape(-1, 28*28).to(device)
        label = labels.to(device)

        outputs = model(image)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Loss
        print(f'Epoch: {epoch + 1} out of {epochs}')
        print(f'Steps: {i+1}/{total_loops}')
        print(f'Loss: {loss.item()}')
        print("-----------------------------------")

# Testing Model Accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
