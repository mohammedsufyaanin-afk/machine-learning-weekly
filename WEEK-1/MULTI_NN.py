import ssl
import torch
import  torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


per_batch = 60
Epochs = 5
change_rate = 0.001

ssl._create_default_https_context = ssl._create_unverified_context
datasets.MNIST.mirrors = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://hf.co/datasets/yannlecun/mnist/resolve/main/'
]

train_dataset = datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)


train_loader = DataLoader(dataset = train_dataset, batch_size = per_batch, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = per_batch, shuffle = True)


class multi(nn.Module):
    def __init__(self):
        super(multi, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = multi()
cross_entropy = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = change_rate)

for epoch in range(Epochs):
    model.train()
    current_loss = 0.0
    for images, labels in train_loader:
        output = model(images)
        loss = cross_entropy(output, labels)

        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        current_loss += loss.item() * images.size(0) #loss in each batch * number of images in the batch(60)

    epoch_loss = current_loss / len(train_loader.dataset) #the value of loss for each epoch(the value of the cost fuction)
    print(f'Epoch [{epoch + 1}/{Epochs}], Loss: {epoch_loss:.4f}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')