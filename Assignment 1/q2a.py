import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load ResNet-18 model pre-trained on ImageNet
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Define a function to extract features from the last or second-last fully connected layer
def extract_features(image, model):
    with torch.no_grad():
        image = image.unsqueeze(0) # add batch dimension
        output = model(image)
        features = output.flatten() # flatten the features from the last or second-last fully connected layer
    return features.numpy()

# Load train and test images and convert them to PyTorch tensors
train_folder = 'D:\\IISC SEM 2\\New folder\\AIP ASSIGNMENTS\\Ruchir Assignment 1\\classification_dataset\\train'
test_folder = 'D:\\IISC SEM 2\\New folder\\AIP ASSIGNMENTS\\Ruchir Assignment 1\\classification_dataset\\test'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = torchvision.datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_folder, transform=transform)

# Extract features from train and test images
train_features = []
train_labels = []
for image, label in train_dataset:
    features = extract_features(image, model)
    train_features.append(features)
    train_labels.append(label)
test_features = []
test_labels = []
for image, label in test_dataset:
    features = extract_features(image, model)
    test_features.append(features)
    test_labels.append(label)

# Train k-NN classifier using the extracted features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)

# Evaluate the k-NN classifier on the test data
accuracy = knn.score(test_features, test_labels)
print('Accuracy on test images:', 100*accuracy)


