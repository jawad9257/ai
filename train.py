import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import glob
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Custom dataset class
class GenderDataset(Dataset):

    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform
  

    def __len__(self):
        return len(self.image_files)
  

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))

        # Convert BGR to RGB since PyTorch works with RGB images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)  # Apply transformations here

        return img, label

# Parameters
epochs = 100
batch_size = 64
img_dims = (96, 96)
lr = 1e-3 # 0.001


# Define the model

class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 1024)  # Adjusted for image size after conv layers
        self.fc2 = nn.Linear(1024, 2)  # Binary classification (male/female)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL image
    transforms.ToTensor(),  # Convert image to PyTorch tensor [C, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Pixel values normalize(mean or standard deviation k according)
])


# Load your dataset
image_files = [f for f in glob.glob(r"C:\AI Project\Gender-Detection-master\Gender-Detection\gender_dataset_face" + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

labels = []
for img in image_files:
    label = img.split(os.path.sep)[-2]
    labels.append(1 if label == "woman" else 0)

# Split data
trainX, testX, trainY, testY = train_test_split(image_files, labels, test_size=0.2, random_state=42)

# Create Dataset and DataLoader
train_dataset = GenderDataset(trainX, trainY, transform=transform)
test_dataset = GenderDataset(testX, testY, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model, Loss, and Optimizer
model = GenderModel()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {correct/total*100}%")

# Save the model
torch.save(model.state_dict(), "gender_model.pth")

# Evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct/total*100}%")
