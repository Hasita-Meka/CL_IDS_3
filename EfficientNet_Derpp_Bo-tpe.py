import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import optuna  # For BO-TPE
from torchvision import datasets, transforms

# Define constants
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256  # Set the batch size to 1024
TRAIN_SPLIT = 0.6  # Use 80% of the data for training, 20% for testing

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = 'Dataset'  # The directory containing all the images

# Load the entire dataset
full_dataset = ImageFolder(data_dir, transform=transform)

# Split the dataset into training and test sets
train_size = int(TRAIN_SPLIT * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset, [train_size, test_size])

# Loaders for training and testing data
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load EfficientNet model
model = efficientnet_b0(pretrained=True)
num_classes = len(full_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify the final layer
model = model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()


# Define Buffer and Derpp class (remains unchanged)

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

    def get_data(self, minibatch_size, device):
        idx = np.random.choice(len(self.data), minibatch_size, replace=False)
        buf_inputs = torch.stack([self.data[i] for i in idx]).to(device)
        buf_labels = torch.tensor([self.labels[i] for i in idx]).to(device)
        buf_logits = torch.stack([self.logits[i] for i in idx]).to(device)
        return buf_inputs, buf_labels, buf_logits


class Derpp:
    """Continual learning via Dark Experience Replay++."""

    def __init__(self, model, criterion, optimizer, buffer_size=1000, alpha=0.5, beta=0.5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.buffer = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward(retain_graph=True)
        tot_loss = loss.item()

        # Dark Experience Replay
        if self.buffer:
            buf_inputs, buf_labels, buf_logits = zip(*self.buffer)
            buf_inputs = torch.stack(buf_inputs).to(DEVICE)
            buf_labels = torch.tensor(buf_labels).to(DEVICE)
            buf_logits = torch.stack(buf_logits).to(DEVICE)

            # MSE Loss on buffered logits
            buf_outputs = self.model(buf_inputs)
            loss_mse = self.alpha * nn.MSELoss()(buf_outputs, buf_logits)
            loss_mse.backward(retain_graph=True)
            tot_loss += loss_mse.item()

            # Cross-Entropy Loss on buffered labels
            loss_ce = self.beta * self.criterion(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.optimizer.step()

        # Store the current inputs, labels, and logits in the buffer
        self.buffer.extend(zip(not_aug_inputs, labels.cpu().numpy(), outputs.data.cpu()))

        return tot_loss


# Training function with Derpp (remains the same)
def train_model_with_derpp(model, train_loader, criterion, optimizer, num_epochs, buffer_size=1000, alpha=0.5,
                           beta=0.5):
    derpp = Derpp(model, criterion, optimizer, buffer_size=buffer_size, alpha=alpha, beta=beta)
    model.train()
    train_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = derpp.observe(images, labels, images)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss


# Evaluation function (remains the same)
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


# Optuna objective function for BO-TPE (remains the same)
def objective(trial):
    # Hyperparameter space
    LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    ALPHA = trial.suggest_uniform('alpha', 0.1, 1.0)
    BETA = trial.suggest_uniform('beta', 0.1, 1.0)

    # DataLoader with fixed batch size of 1024
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_loss = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS, buffer_size=1000,
                                        alpha=ALPHA, beta=BETA)

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    # Optuna will minimize the loss
    return avg_loss


# Run the BO-TPE optimization (remains the same)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Start timing
start_time = time.time()

# Train and evaluate using the best hyperparameters
LEARNING_RATE = study.best_params['learning_rate']
ALPHA = study.best_params['alpha']
BETA = study.best_params['beta']

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model using Derpp
train_loss = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS, buffer_size=1000,
                                    alpha=ALPHA, beta=BETA)

# Evaluate the model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Print performance metrics
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print(f'Confusion Matrix:\n {cm}')
print(f'Time taken: {time_taken:.2f} seconds')

# Plot training loss over epochs
plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()
