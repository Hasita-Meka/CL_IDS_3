import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import mobilenet_v2  # Import MobileNetV2
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torch.nn.functional as F
import optuna  # Import Optuna for hyperparameter optimization
import json  # Import JSON for saving hyperparameters
from pathlib import Path  # Import Path for file operations

# Define constants
BATCH_SIZE = 256  # Batch size for training
NUM_EPOCHS = 30
TRAIN_RATIO = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_folder = 'output_8'  # Define output folder
os.makedirs(output_folder, exist_ok=True)

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset_dir = 'Dataset'  # Update to your single folder path
dataset = ImageFolder(dataset_dir, transform=transform)

# Split dataset into training and testing
train_size = int(len(dataset) * TRAIN_RATIO)
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# Load MobileNetV2 model
def create_model(learning_rate):
    model = mobilenet_v2(weights='DEFAULT')
    num_classes = len(train_data.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify the final layer
    model = model.to(DEVICE)
    return model, optim.Adam(model.parameters(), lr=learning_rate)


# Loss function
criterion = nn.CrossEntropyLoss()


class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data = []
        self.labels = []
        self.logits = []

    def is_empty(self):
        return len(self.data) == 0

    def add_data(self, examples, labels, logits):
        self.data.extend(examples)
        self.labels.extend(labels)
        self.logits.extend(logits)

        if len(self.data) > self.buffer_size:
            self.data = self.data[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]
            self.logits = self.logits[-self.buffer_size:]

    def get_data(self, minibatch_size, transform, device):
        idx = np.random.choice(len(self.data), minibatch_size, replace=False)
        buf_inputs = torch.stack([self.data[i] for i in idx]).to(device)
        buf_labels = torch.tensor([self.labels[i] for i in idx]).to(device)
        buf_logits = torch.stack([self.logits[i] for i in idx]).to(device)
        return buf_inputs, buf_labels, buf_logits


class DerppModel:
    def __init__(self, model, learning_rate, alpha=0.1, beta=0.1, buffer_size=100):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.buffer = Buffer(buffer_size)
        self.opt = optim.Adam(model.parameters(), lr=learning_rate)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, transform, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # DER++ MSE loss
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            # DER++ Cross Entropy loss
            buf_outputs = self.model(buf_inputs)
            loss_ce = self.beta * criterion(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()
        self.buffer.add_data(not_aug_inputs, labels, outputs.detach())

        return tot_loss


# Function to train the model using DERPP
def train_model_with_derpp(model, train_loader, criterion, optimizer, num_epochs, buffer_size=1000, alpha=0.5,
                           beta=0.5):
    derpp = DerppModel(model, optimizer.param_groups[0]['lr'], alpha=alpha, beta=beta, buffer_size=buffer_size)
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            not_aug_images = images.clone()  # Store original images for DER++
            loss = derpp.observe(images, labels, not_aug_images)
            epoch_loss += loss

            # Calculate accuracy
            outputs = derpp.model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_loss.append(avg_loss)
        train_accuracy.append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return train_loss, train_accuracy


# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, precision, recall, auc_roc, cm


# Function to plot loss vs accuracy curves
def plot_metrics(train_loss, accuracy_list):
    epochs = range(1, len(train_loss) + 1)

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_folder, 'train_loss_plot.png'))
    plt.close()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_list, 'g', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_folder, 'train_accuracy_plot.png'))
    plt.close()


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ALPHA = trial.suggest_float('alpha', 0.1, 1.0)
    BETA = trial.suggest_float('beta', 0.1, 1.0)

    # Load datasets with the suggested batch size
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create model with the suggested learning rate
    model, optimizer = create_model(learning_rate)

    # Train the model
    train_loss, train_accuracy = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS,
                                                        buffer_size=1000, alpha=0.5, beta=0.5)

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    return accuracy  # We want to maximize accuracy


# Start the hyperparameter optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

best_hyperparams_file = os.path.join(output_folder, 'best_hyperparameters.json')
with open(best_hyperparams_file, 'w') as f:
    json.dump(study.best_params, f)

# Start timing
start_time = time.time()

# Train the model with the best hyperparameters
LEARNING_RATE = study.best_params['learning_rate']
ALPHA = study.best_params['alpha']
BETA = study.best_params['beta']

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model using Derpp
train_loss, train_accuracy = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS,
                                                    buffer_size=1000, alpha=ALPHA, beta=BETA)

# Evaluate the model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

# Stop timing
end_time = time.time()
runtime = end_time - start_time

# Save the model's state dictionary
model_weights_file = os.path.join(output_folder, 'model_weights.pth')
torch.save(model.state_dict(), model_weights_file)

# Save training loss
train_loss_file = os.path.join(output_folder, 'train_loss.json')
with open(train_loss_file, 'w') as f:
    json.dump(train_loss, f)

# Save results to JSON file
results = {
    "avg_loss": avg_loss,
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "auc_roc": auc_roc,
    "confusion_matrix": cm.tolist(),
    "runtime": runtime
}

with open(os.path.join(output_folder, 'results.json'), 'w') as f:
    json.dump(results, f)

# Plot metrics
plot_metrics(train_loss, train_accuracy)