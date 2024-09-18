import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Load ResNet-18 model pre-trained on ImageNet
model = torchvision.models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Define a new classifier for the final fully connected layer
model.fc = nn.Linear(512, 6) 

# Load the training and test data
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = torchvision.datasets.ImageFolder(root='D:\\IISC SEM 2\\New folder\\AIP ASSIGNMENTS\\Ruchir Assignment 1\\classification_dataset\\train', transform=transform)
test_data = torchvision.datasets.ImageFolder(root='D:\\IISC SEM 2\\New folder\\AIP ASSIGNMENTS\\Ruchir Assignment 1\\classification_dataset\\test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the classifier on the training data
for epoch in range(6):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the performance of the classifier on the test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
