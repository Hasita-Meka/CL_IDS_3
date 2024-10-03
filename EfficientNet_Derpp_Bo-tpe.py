import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import numpy as np
from collections import deque
import optuna  # For BO-TPE
import json
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define constants
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
TRAIN_RATIO = 0.6

# Save everything to the output folder
output_folder = 'output_7'
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

# Load EfficientNet model
model = efficientnet_b0(weights='DEFAULT')
num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify the final layer
model = model.to(DEVICE)

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


# Training function with Derpp
def train_model_with_derpp(model, train_loader, criterion, optimizer, num_epochs, buffer_size=1000, alpha=0.5,
                           beta=0.5):
    derpp = Derpp(model, criterion, optimizer, buffer_size=buffer_size, alpha=alpha, beta=beta)
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = derpp.observe(images, labels, images)
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


# Evaluation function
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


# Optuna objective function for BO-TPE
def objective(trial):
    # Hyperparameter space
    LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ALPHA = trial.suggest_float('alpha', 0.1, 1.0)
    BETA = trial.suggest_float('beta', 0.1, 1.0)

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


# Run the BO-TPE optimization
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best hyperparameters: ", study.best_params)

best_hyperparams_file = os.path.join(output_folder, 'best_hyperparameters.json')
with open(best_hyperparams_file, 'w') as f:
    json.dump(study.best_params, f)

# Start timing
start_time = time.time()

# Train and evaluate using the best hyperparameters
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
print(f'Time Taken: {time_taken:.2f} seconds')
print('Confusion Matrix:\n', cm)

# Save the model's state dictionary
model_weights_file = os.path.join(output_folder, 'model_weights.pth')
torch.save(model.state_dict(), model_weights_file)

# Save training loss
train_loss_file = os.path.join(output_folder, 'train_loss.json')
with open(train_loss_file, 'w') as f:
    json.dump(train_loss, f)

# Save evaluation metrics
metrics_file = os.path.join(output_folder, 'evaluation_metrics.json')
metrics = {
    "avg_loss": avg_loss,
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "auc_roc": auc_roc,
    "time_taken": time_taken,
    "confusion_matrix": cm.tolist()  # Convert numpy array to list for JSON serialization
}
with open(metrics_file, 'w') as f:
    json.dump(metrics, f)

plt.figure()
plt.plot(train_accuracy)
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(output_folder, 'train_accuracy_plot.png'))
plt.close()

plt.figure()
plt.plot(train_loss)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_folder, 'train_loss_plot.png'))
plt.close()

print(f'Saved model weights to {model_weights_file}')
print(f'Saved training loss to {train_loss_file}')
print(f'Saved evaluation metrics to {metrics_file}')
print(f'Saved training accuracy plot to {os.path.join(output_folder, "train_accuracy_plot.png")}')
print(f'Saved training loss plot to {os.path.join(output_folder, "train_loss_plot.png")}')
