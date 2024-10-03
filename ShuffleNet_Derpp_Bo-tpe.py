import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import shufflenet_v2_x0_5
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import optuna  # Add this import for Optuna
from torchvision import datasets

# Define constants
BATCH_SIZE = 256
NUM_EPOCHS = 30
TRAIN_RATIO = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = 1000  # Define buffer size
ALPHA = 0.5  # MSE loss weight
BETA = 1.0  # Cross-entropy loss weight
# Save everything to the output folder
output_folder = 'output_9'
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


# Load ShuffleNet model
def create_model(learning_rate):
    model = shufflenet_v2_x0_5(weights='DEFAULT')
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# Loss function
criterion = nn.CrossEntropyLoss()


# Buffer class for experience replay
class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.inputs = []
        self.labels = []
        self.logits = []

    def add_data(self, inputs, labels, logits):
        # Add new data to the buffer
        for i in range(len(inputs)):
            if len(self.inputs) < self.buffer_size:
                self.inputs.append(inputs[i].cpu())
                self.labels.append(labels[i].cpu())
                self.logits.append(logits[i].cpu())
            else:
                # If the buffer is full, replace randomly
                idx = np.random.randint(0, self.buffer_size)
                self.inputs[idx] = inputs[i].cpu()
                self.labels[idx] = labels[i].cpu()
                self.logits[idx] = logits[i].cpu()

    def get_data(self, batch_size, transform, device):
        # Fetch random data from the buffer
        indices = np.random.choice(len(self.inputs), batch_size, replace=False)
        buffer_inputs = torch.stack([self.inputs[i] for i in indices]).to(device)
        buffer_labels = torch.tensor([self.labels[i] for i in indices]).to(device)
        buffer_logits = torch.tensor(np.stack([self.logits[i] for i in indices])).to(device)

        # Apply transform if needed (make sure transform is compatible with Tensors)
        return buffer_inputs, buffer_labels, buffer_logits

    def is_empty(self):
        # Check if buffer is empty
        return len(self.inputs) == 0


# Derpp model for continual learning
class Derpp:
    def __init__(self, model, buffer_size, alpha, beta):
        self.model = model
        self.buffer = Buffer(buffer_size)
        self.alpha = alpha
        self.beta = beta

    def observe(self, inputs, labels, not_aug_inputs, optimizer):
        # Forward pass for current data
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)  # Retain graph for further backward pass
        tot_loss = loss

        # Experience Replay with buffer data
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, transform, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # MSE loss on logits
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward(retain_graph=True)  # Retain graph here as well
            tot_loss += loss_mse

            # Cross-entropy loss on labels
            loss_ce = self.beta * criterion(buf_outputs, buf_labels)
            loss_ce.backward()  # No need to retain graph after the last backward call
            tot_loss += loss_ce

        optimizer.step()

        # Add data to buffer
        self.buffer.add_data(not_aug_inputs, labels, outputs.data)
        return tot_loss


# Function to train the model using Derpp
def train_model_with_derpp(model, train_loader, criterion, optimizer, num_epochs, buffer_size=1000, alpha=0.5,
                           beta=0.5):
    derpp = Derpp(model, buffer_size, alpha, beta)
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = derpp.observe(images, labels, images, optimizer)  # Add not_aug_inputs here
            epoch_loss += loss.item()

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


# Optimization function for Optuna
def objective(trial):
    # Hyperparameter optimization
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    beta = trial.suggest_float('beta', 0.1, 1.0)

    model = create_model(learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Train the model using Derpp
    train_loss, train_accuracy = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS,
                                                        buffer_size=BUFFER_SIZE, alpha=ALPHA, beta=BETA)

    if train_accuracy > study.best_value:
        study.best_value = train_accuracy
        best_model = model

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    return accuracy


start_time = time.time()

# Start the Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # You can increase the number of trials for better results

# Print the best hyperparameters found
print("Best hyperparameters: ", study.best_params)

best_hyperparams_file = os.path.join(output_folder, 'best_hyperparameters.json')
with open(best_hyperparams_file, 'w') as f:
    json.dump(study.best_params, f)

# After hyperparameter tuning, you can retrain the model with the best hyperparameters if desired
LEARNING_RATE = study.best_params['learning_rate']
ALPHA = study.best_params['alpha']
BETA = study.best_params['beta']

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss, train_accuracy = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS,
                                                    buffer_size=1000, alpha=ALPHA, beta=BETA)

# Evaluate the best model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(best_model, test_loader)

end_time = time.time()
time_taken = end_time - start_time

# Save the model's state dictionary
model_weights_file = os.path.join(output_folder, 'model_weights.pth')
torch.save(best_model.state_dict(), model_weights_file)

# Save training loss
train_loss_file = os.path.join(output_folder, 'train_loss.json')
with open(train_loss_file, 'w') as f:
    json.dump(train_loss, f)

# Save training accuracy
train_accuracy_file = os.path.join(output_folder, 'train_accuracy.json')
with open(train_accuracy_file, 'w') as f:
    json.dump(train_accuracy, f)

# Save evaluation results
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
plt.plot(train_loss)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_folder, 'train_loss_plot.png'))
plt.close()

plt.figure()
plt.plot(train_accuracy)
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(output_folder, 'train_accuracy_plot.png'))
plt.close()

print(f'Saved model weights to {model_weights_file}')
print(f'Saved training loss to {train_loss_file}')
print(f'Saved evaluation metrics to {metrics_file}')
print(f'Saved training accuracy plot to {os.path.join(output_folder, "train_accuracy_plot.png")}')
print(f'Saved training loss plot to {os.path.join(output_folder, "train_loss_plot.png")}')