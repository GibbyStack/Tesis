'''
CNN --> C-P-C-P-F

0) Prepare data
    --> MNIST - FashionMNIST
        --> 28x28 gray scale
        --> 60,000 training examples
        --> 10,000 test examples
        --> 10 classes
    --> CIFAR 10
        --> 32x32 color images
        --> 50,000 training examples
        --> 10,000 test examples
        --> 10 classes
    --> CIFAR 100
        --> 100 classes
1) Design model
2) Construct loss and optimizer
3) Training loop
4) Testing model
'''

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/conv-cifar10')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 40
batch_size = 100
learning_rate = 0.001

# 0) Prepare data
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                  train=True,
                                                  transform=transform.ToTensor(),
                                                  download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=False,
                                                 transform=transform.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# Add images to tensorboard
examples = iter(test_loader)
images, labels = examples._next_data()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('cifar10_images', img_grid)
writer.close()

# for i in range(batch_size):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(images[i][0], cmap='gray')
# plt.tight_layout()
# plt.show()

# 1) Desing model
class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [MNIST y FASHIONM --> 7x7x20 = 980] [CIFAR10 --> 8x8x20 = 1280]
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [CIFAR10 --> 8x8x40=2560]
        self.fc1 = nn.Linear(in_features=2560, out_features=1280)
        self.fc2 = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        
        # Flatten - Matrix --> vector
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
# Add model graph to tensorboard
model = CNN(num_classes=10).to(device)
writer.add_graph(model, images)
writer.close()

# 2) Construct loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) Training loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
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

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch: [{epoch+1}/{num_epochs}], step: [{i+1}/{n_total_steps}], loss = {loss.item():.4f}')
            # Add scalar loss and training acc to Tensorboard
            writer.add_scalar('Training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('Accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
writer.close()

# Testing model
class_labels = []
class_preds = []
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

        # Important - Predictions == probabilities
        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        class_preds.append(class_probs_batch)
        class_labels.append(labels)
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')

    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    # # Add precision --> (PPV - Curve Roc) to Tensorboard
    # fig, ax = plt.subplots(figsize=(6, 6))
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    #     RocCurveDisplay.from_predictions(
    #         labels_i,
    #         preds_i,
    #         name=f'ROC curve for {str(i)}',
    #         color=f'C{str(i)}',
    #         ax = ax
    #     )
    # plt.title('ROC curve to One-vs-Rest multiclass')
    # plt.legend()
    # plt.show()
