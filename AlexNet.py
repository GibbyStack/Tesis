'''
AlexNet --> C-R-N-P-C-R-N-P-C-R-C-R-C-R-P-F-R-F-R-F-R
C - Convolucion
R - ReLU
N - Normalization
P - Pooling
F - Fully conected

0) Prepare data
    --> CIFAR 100
        --> 32x32 color images -> 224x244x3
        --> 50,000 training examples
        --> 10,000 test examples
        --> 100 classes
1) Design model
    C => 96 kernels = 11x11x3, stride = 4
    P => size = 3, stride = 2
    C => 256 kernels = 5x5x48, stride = 1
    P => size = 3, stride = 2
    C => 384 kernels = 3x3x256, stride = 1
    C => 384 kernels = 3x3x192, stride = 1
    C => 256 kernels = 3x3x192, stride = 1
    P => size = 3, stride = 2
    F => in = 4096, out = 4096
    F => in = 4096, out = 4096
    F => in = 4096, out = 100
2) Construct loss and optimizer
    --> Cross Entropy Loss
    --> Stochastic Gradient Descent
        --> momentum = 0.9
        --> weight decay = 0.0005
3) Training loop
    --> Batch size = 128
    --> Learning rate = 0.01
    --> Number epochs = 90
4) Testing model
'''

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/conv-cifar100')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameter
num_epochs = 20
batch_size = 128
learning_rate = 0.01

# 0) Prepare data
train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor()
                                              ]),
                                              download=True)

test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor()
                                              ]))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# Get first batch of images
examples = iter(test_loader)
images, labels = examples._next_data()

# Add images to tensorboard
img_grid = torchvision.utils.make_grid(images)
writer.add_image('Cifar100_images', img_grid)
writer.close()

# 1) Design model
class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # Flatten - Matrix --> vector
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Add model graph to tensoroard
model = AlexNet(num_classes=100).to(device)
writer.add_graph(model, images.to(device))
writer.close()

# 2) Construct loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

# 3) Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Backward pass
        loss = cost(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad() # Vaciar gradientes


        if (i+1) % 100 == 0:
            running_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct = (predicted == labels).sum().item()
            print(f'Epoch: [{epoch+1}/{num_epochs}], step: [{i+1}/{n_total_steps}], loss = {loss.item():.4f}')
            # Add scalar loss and training acc to Tensorboard
            writer.add_scalar('Training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('Accuracy', running_correct / 100, epoch * n_total_steps + i)

# Testing model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1) # (value, index)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')