import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
#from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np

# Set random seeds for CPU
torch.manual_seed(33)
random.seed(33)
np.random.seed(33)

# Set random seeds for GPU (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(33)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the path to your CSV file
PATH_CSV_TRAIN = 'train.csv'
PATH_CSV_VAL = 'val.csv'
DIR_IMG = './'
NB_EPOCHS = 10
BATCH_SIZE = 8

"""
        T.Resize(image_size + 4),
        T.CenterCrop(image_size),
        T.RandomRotation(40),
        T.RandomAffine(
            degrees=10,
            translate=(0.01, 0.12),
            shear=(0.01, 0.03),
        ),
        T.RandomHorizontalFlip(), 
        T.RandomVerticalFlip(),
"""

# Define the image transformation pipeline for training data
train_transform = transforms.Compose([
    transforms.Resize((366, 366)),  # Resize the input image
    transforms.RandomCrop((320, 320)),  # Randomly crop the image
    # transforms.RandomRotation(40),
    # transforms.RandomAffine(
    #     degrees=10,
    #     translate=(0.01, 0.12),
    #     shear=(0.01, 0.03)
    #     ),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    # transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Define the image transformation pipeline for validation data
val_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize the input image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create a custom dataset class
class CSV_Dataset(Dataset):
    def __init__(self, path_csv_file, transform=None):
        self.data = pd.read_csv(path_csv_file)
        self.transform = transform
        self.classes = sorted(list(self.data['label'].unique()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(DIR_IMG,self.data.iloc[idx]['filepath'])
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label

# Create an instance of the custom dataset for training and validation
train_dataset = CSV_Dataset(PATH_CSV_TRAIN, transform=train_transform)
val_dataset = CSV_Dataset(PATH_CSV_VAL, transform=val_transform)

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Load the pre-trained ResNet model
model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
#print(model.classifier)
num_classes = len(train_dataset.classes)

# # Replace the last fully connected layer with a new one suitable for the number of classes
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)

# Replace the last fully connected layer with a new one suitable for the number of classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)

#learnable_modules = ['features.5.2', 'features.6', 'features.7', 'features.8', 'classifier']
# learnable_modules = ['classifier']
# model.requires_grad_(False)
# modules = dict(model.named_modules())
# for name in learnable_modules:
#     modules[name].requires_grad_(True)

# Define the loss function and optimizer
#criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

val_loss_best = 9999.99

for epoch in range(NB_EPOCHS):
    print(f'Epoch: {epoch}')
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    print(f'Train loss: {train_loss:.5f}') 

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    print(f'Val loss: {val_loss:.5f} - Val acc: {val_accuracy:.5f}')

    if val_loss < val_loss_best:
        val_loss_best = val_loss
        # Save the model weights
        path_weights = f'weights/efficientnet_b4.pth'
        torch.save(model.state_dict(), path_weights)
        print(f"Weights saved to {path_weights}!")
os.rename(path_weights, os.path.splitext(path_weights)[0]+f'_val_loss_{val_loss_best:.5f}.pth')