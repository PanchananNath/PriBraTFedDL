"""
brain_tumor_federated_learning.py is the python code developed by Er. Panchanan Nath for simulating federated learning using 5 deep learning model on brain tumor classification dataset.


Implements 5 custom models: SimpleCNN, VGGLike, ResNetLike, SimpleDenseNetLike, MobileNetLite-like.

Key changes for Federated Learning:
 - Data partitioning into client datasets.
 - Federated Averaging (FedAvg) implementation for model aggregation.
 - Simulation of client-server communication rounds.
 - Each model is trained in a federated manner.

Usage:
 - Set DATA_DIR variable to root folder containing Training/ and Testing/
 - Run on GPU-enabled machine / Colab
"""

import os
import random
import time
import copy
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, utils

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_DIR = "./"   # root containing Training/ and Testing/
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR  = os.path.join(DATA_DIR, "Testing")

OUTPUT_DIR = "./results_brain_tumor_federated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 4  # pituitary, notumor, meningioma, glioma
BATCH_SIZE = 32
# NUM_EPOCHS is now 'num_communication_rounds' for federated learning
NUM_COMMUNICATION_ROUNDS = 100 # Number of global aggregation rounds
CLIENT_LOCAL_EPOCHS = 10 # Number of epochs each client trains locally
NUM_CLIENTS = 10 # Number of simulated clients
CLIENT_FRACTION = 0.5 # Fraction of clients participating in each round

IMAGE_SIZE = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
STEP_LR_STEP = 7
STEP_LR_GAMMA = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Reproducibility
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------------------------
# Data transforms & loaders
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

try:
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=test_transform)
except Exception as e:
    print(f"Error loading datasets. Please check DATA_DIR and folder structure. Error: {e}")
    print(f"Expected TRAIN_DIR: {TRAIN_DIR}")
    print(f"Expected TEST_DIR: {TEST_DIR}")
    exit() # Exit if data loading fails

if not full_train_dataset.classes:
    print("Error: No classes found in the training dataset. Ensure your data structure is correct (e.g., Training/class1/img.jpg).")
    exit()

class_names = full_train_dataset.classes
NUM_CLASSES = len(class_names) # Ensure NUM_CLASSES matches actual detected classes
print("Class names:", class_names)

# Create a small validation split from the training dataset (for server-side evaluation)
val_ratio = 0.1
num_full_train = len(full_train_dataset)
if num_full_train == 0:
    print("Error: Training dataset is empty. Please provide images in the Training folder.")
    exit()

indices = list(range(num_full_train))
split = int(np.floor(val_ratio * num_full_train))
random.shuffle(indices)
train_indices_for_clients, val_indices_server = indices[split:], indices[:split]

# Create server-side validation loader
val_dataset_server = Subset(full_train_dataset, val_indices_server)
val_loader_server = DataLoader(val_dataset_server, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Total train samples for clients: {len(train_indices_for_clients)}")
print(f"Server-side Val samples: {len(val_indices_server)}")
print(f"Test samples: {len(test_dataset)}")

# Partition data for clients
def distribute_data_to_clients(dataset, num_clients):
    data_indices = list(range(len(dataset)))
    random.shuffle(data_indices)
    # Simple equal distribution; in real FL, data distribution can be uneven
    client_data_splits = np.array_split(data_indices, num_clients)
    client_datasets = []
    for split_indices in client_data_splits:
        # Only create a Subset if the split_indices are not empty
        if split_indices.size > 0:
            client_datasets.append(Subset(dataset, split_indices.tolist()))
        else:
            client_datasets.append(Subset(dataset, [])) # Append an empty subset if no data for this client
    return client_datasets

print(f"Distributing data to {NUM_CLIENTS} clients...")
client_train_datasets = distribute_data_to_clients(Subset(full_train_dataset, train_indices_for_clients), NUM_CLIENTS)
client_train_loaders = []
for ds in client_train_datasets:
    if len(ds) > 0:
        client_train_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS, pin_memory=True))
    else:
        # Append an empty loader or handle appropriately if client has no data
        client_train_loaders.append(None) # Mark as None if client has no data for this example

print(f"First client has {len(client_train_datasets[0]) if client_train_datasets and len(client_train_datasets[0]) > 0 else 0} samples.")


# ---------------------------
# Utility functions for visualization, metrics, saving
# ---------------------------
def imshow_tensor(img_tensor, title=None, mean=None, std=None):
    # img_tensor: CxHxW
    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None: std = [0.229, 0.224, 0.225]
    img = img_tensor.numpy().transpose((1,2,0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')

def save_history_csv(history, filepath):
    # history: dict with lists
    keys = list(history.keys())
    rows = list(zip(*[history[k] for k in keys]))
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)

def plot_history(history, model_name, outdir):
    epochs = np.arange(1, len(history['global_val_loss'])+1) # Now using global_val_loss/acc
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['global_train_loss'], label='global_train_loss')
    plt.plot(epochs, history['global_val_loss'],   label='global_val_loss')
    plt.xlabel('Communication Round'); plt.ylabel('Loss'); plt.title(f"{model_name} Federated Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['global_train_acc'], label='global_train_acc')
    plt.plot(epochs, history['global_val_acc'],   label='global_val_acc')
    plt.xlabel('Communication Round'); plt.ylabel('Accuracy'); plt.title(f"{model_name} Federated Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_federated_history.png"))
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, outpath=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.title(title)
    if outpath:
        plt.savefig(outpath)
    plt.show()

# ---------------------------
# Model definitions (custom) - same as original
# ---------------------------

# 1) Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #56
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #28
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2) VGG-like small
class VGGLike(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGGLike, self).__init__()
        def conv_block(in_c, out_c, n=2):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_c),
                           nn.ReLU(inplace=True)]
                in_c = out_c
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            return nn.Sequential(*layers)
        self.features = nn.Sequential(
            conv_block(3, 32, n=2),   # 112
            conv_block(32, 64, n=2),  # 56
            conv_block(64, 128, n=2), # 28
            conv_block(128, 256, n=2) # 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3) ResNet-like small (BasicBlock)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2], num_classes=NUM_CLASSES):
        super(ResNetLike, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # ->112
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # ->56
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# 4) Simple DenseNet-like (not full DenseNet - a small "dense block" style)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, n_layers=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            channels += growth_rate
        self.out_channels = channels
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features,1)

class SimpleDenseNetLike(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleDenseNetLike, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #112
        self.pool = nn.MaxPool2d(3, stride=2, padding=1) #56
        self.db1 = DenseBlock(64, growth_rate=32, n_layers=3)
        self.trans1 = nn.Sequential(nn.BatchNorm2d(self.db1.out_channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(self.db1.out_channels, 128, kernel_size=1), nn.AvgPool2d(2))
        self.db2 = DenseBlock(128, growth_rate=32, n_layers=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.db2.out_channels, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.db1(x)
        x = self.trans1(x)
        x = self.db2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# 5) MobileNet-like light (depthwise separable convs)
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetLite(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MobileNetLite, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 32, 2), #112
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            *[conv_dw(512,512,1) for _ in range(2)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)


# ---------------------------
# Training & evaluation utilities (adapted for federated learning)
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    # Check if dataloader is not None and has items
    if dataloader is None or len(dataloader.dataset) == 0:
        return 0.0, 0.0 # Return 0 loss/acc if no data to train on

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def eval_model(model, dataloader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_outputs = []
    all_preds = []
    all_labels = []

    if dataloader is None or len(dataloader.dataset) == 0:
        if return_preds:
            return 0.0, 0.0, torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            return 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    if return_preds:
        return epoch_loss, epoch_acc, torch.cat(all_outputs), torch.cat(all_preds), torch.cat(all_labels)
    else:
        return epoch_loss, epoch_acc

def federated_averaging(global_model, client_models, client_data_sizes):
    """
    Aggregates client model weights using Federated Averaging (FedAvg).
    client_data_sizes: list of number of samples for each participating client.
    """
    # Filter out clients that did not participate or had empty data (None loaders)
    active_client_models = [model for model in client_models if model is not None]
    active_client_data_sizes = [size for size, model in zip(client_data_sizes, client_models) if model is not None]

    if not active_client_models:
        # No active clients participated or all had empty data, return current global model state
        print("Warning: No active client models received for aggregation. Returning current global model state.")
        return global_model.state_dict()

    total_data_size = sum(active_client_data_sizes)
    if total_data_size == 0:
        # All participating active clients had zero data, cannot average meaningfully
        print("Warning: Sum of data sizes from active participating clients is zero. Returning current global model state.")
        return global_model.state_dict()

    averaged_weights = {}
    # Initialize averaged weights with the first active client's weights, scaled
    # Ensure active_client_models[0] actually exists and has parameters
    try:
        for name, param in active_client_models[0].state_dict().items():
            averaged_weights[name] = param.data.clone() * (active_client_data_sizes[0] / total_data_size)

        # Add weighted weights from remaining active clients
        for i in range(1, len(active_client_models)):
            client_weight_factor = active_client_data_sizes[i] / total_data_size
            for name, param in active_client_models[i].state_dict().items():
                if name in averaged_weights:
                    averaged_weights[name] += param.data.clone() * client_weight_factor
                else:
                    print(f"Warning: Parameter {name} not found in averaged_weights during aggregation for client {i}. Skipping.")
    except IndexError:
        print("Error: active_client_models is unexpectedly empty during aggregation initialization. Returning current global model state.")
        return global_model.state_dict()
    except Exception as e:
        print(f"An error occurred during federated averaging: {e}. Returning current global model state.")
        return global_model.state_dict()

    return averaged_weights


# ---------------------------
# Federated Learning Training Loop
# ---------------------------
# Initialize models with NUM_CLASSES based on actual detected classes
models_to_train = {
    "SimpleCNN": SimpleCNN(num_classes=NUM_CLASSES),
    "VGGLike": VGGLike(num_classes=NUM_CLASSES),
    "ResNetLike": ResNetLike(num_classes=NUM_CLASSES),
    "SimpleDenseNetLike": SimpleDenseNetLike(num_classes=NUM_CLASSES),
    "MobileNetLite": MobileNetLite(num_classes=NUM_CLASSES),
}

trained_federated_models = {}

# Main execution logic should be wrapped in this block for multiprocessing
if __name__ == "__main__":
    for model_name, global_model in models_to_train.items():
        print("\n" + "="*80)
        print(f"Starting Federated Training for model: {model_name}")
        print("="*80)

        global_model = global_model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        history = defaultdict(list)
        best_global_acc = 0.0
        best_global_model_wts = copy.deepcopy(global_model.state_dict())

        # Initialize client models (they start with the global model's weights)
        client_models_pool = []
        for _ in range(NUM_CLIENTS):
            client_model_instance = type(global_model)(num_classes=NUM_CLASSES) # Create new instance of the same model type
            client_model_instance.load_state_dict(global_model.state_dict()) # Initialize with global weights
            client_models_pool.append(client_model_instance.to(DEVICE))


        for round_num in range(1, NUM_COMMUNICATION_ROUNDS + 1):
            t0 = time.time()
            print(f"\n--- Communication Round {round_num}/{NUM_COMMUNICATION_ROUNDS} ---")

            # 1. Server selects a fraction of clients for this round
            num_participating_clients = max(1, int(NUM_CLIENTS * CLIENT_FRACTION))
            selected_client_indices = random.sample(range(NUM_CLIENTS), num_participating_clients)
            print(f"Selected {num_participating_clients} clients for this round.")

            participating_client_models_for_aggregation = [] # Will store models that actually trained
            participating_client_data_sizes_for_aggregation = [] # Will store data sizes of those models
            round_local_train_losses = []
            round_local_train_accs = []

            # 2. Each selected client trains its local model
            for client_idx in selected_client_indices:
                client_model = client_models_pool[client_idx]
                client_dataloader = client_train_loaders[client_idx]
                client_data_size = len(client_train_datasets[client_idx])

                if client_dataloader is None or client_data_size == 0:
                    print(f"Client {client_idx} has no data, skipping local training for this round.")
                    continue # Skip clients with no data

                # Client receives the current global model weights
                client_model.load_state_dict(global_model.state_dict())
                client_optimizer = optim.Adam(client_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                # No scheduler per client, or a very light one

                # Train locally for CLIENT_LOCAL_EPOCHS
                local_train_loss_sum = 0.0
                local_train_acc_sum = 0.0
                for local_epoch in range(CLIENT_LOCAL_EPOCHS):
                    local_loss, local_acc = train_one_epoch(
                        client_model, client_dataloader, criterion, client_optimizer, DEVICE
                    )
                    local_train_loss_sum += local_loss
                    local_train_acc_sum += local_acc
                
                round_local_train_losses.append(local_train_loss_sum / CLIENT_LOCAL_EPOCHS)
                round_local_train_accs.append(local_train_acc_sum / CLIENT_LOCAL_EPOCHS)

                # Client sends its updated weights to the server
                participating_client_models_for_aggregation.append(client_model)
                participating_client_data_sizes_for_aggregation.append(client_data_size)

            # Calculate average local training metrics for this round
            avg_round_train_loss = np.mean(round_local_train_losses) if round_local_train_losses else 0.0
            avg_round_train_acc = np.mean(round_local_train_accs) if round_local_train_accs else 0.0


            # 3. Server aggregates the weights from participating clients
            # Pass only the models and sizes of clients that actually trained
            aggregated_weights = federated_averaging(
                global_model,
                participating_client_models_for_aggregation,
                participating_client_data_sizes_for_aggregation
            )
            global_model.load_state_dict(aggregated_weights)

            # 4. Server evaluates the global model on a central validation set
            global_val_loss, global_val_acc = eval_model(global_model, val_loader_server, criterion, DEVICE)

            history['global_train_loss'].append(avg_round_train_loss) # Avg client train loss
            history['global_train_acc'].append(avg_round_train_acc)   # Avg client train acc
            history['global_val_loss'].append(global_val_loss)
            history['global_val_acc'].append(global_val_acc)

            if global_val_acc > best_global_acc:
                best_global_acc = global_val_acc
                best_global_model_wts = copy.deepcopy(global_model.state_dict())
                torch.save(global_model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_federated_best.pt"))

            t1 = time.time()
            print(f"Round {round_num}/{NUM_COMMUNICATION_ROUNDS} | Avg Client Train Loss: {avg_round_train_loss:.4f} Avg Client Train Acc: {avg_round_train_acc:.4f} | Global Val Loss: {global_val_loss:.4f} Global Val Acc: {global_val_acc:.4f} | Time: {(t1-t0):.1f}s")


        # Load best global model weights after all rounds
        global_model.load_state_dict(best_global_model_wts)
        # Save final global model
        torch.save(global_model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_federated_final.pt"))

        # Save history CSV and plot
        history_path = os.path.join(OUTPUT_DIR, f"{model_name}_federated_history.csv")
        save_history_csv(history, history_path)
        plot_history(history, model_name, OUTPUT_DIR)

        trained_federated_models[model_name] = global_model
        # also save a small metadata file
        with open(os.path.join(OUTPUT_DIR, f"{model_name}_federated_meta.json"), "w") as f:
            json.dump({"model_name": model_name, "best_global_val_acc": best_global_acc,
                       "communication_rounds": NUM_COMMUNICATION_ROUNDS, "num_clients": NUM_CLIENTS}, f, indent=2)

    # ---------------------------
    # Evaluate models on test set & save metrics (using the final aggregated models)
    # ---------------------------
    def evaluate_and_report(model, loader, device, classes):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        cm = confusion_matrix(all_labels, all_preds)
        cls_report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
        p_r_f = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(classes)))
        accuracy = (all_preds == all_labels).mean()
        return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": cls_report, "prf": p_r_f, "preds": all_preds, "labels": all_labels, "probs": all_probs}

    overall_federated_results = {}
    for model_name, model in trained_federated_models.items():
        print("\n" + "-"*60)
        print("Testing Federated Model:", model_name)
        print("-"*60)
        res = evaluate_and_report(model.to(DEVICE), test_loader, DEVICE, class_names)
        overall_federated_results[model_name] = res
        # Save classification report
        with open(os.path.join(OUTPUT_DIR, f"{model_name}_federated_classification_report.txt"), "w") as f:
            f.write(f"Test Accuracy: {res['accuracy']:.4f}\n\n")
            f.write(res['classification_report'])
        # Save confusion matrices plots
        plot_confusion_matrix(res['confusion_matrix'], class_names, normalize=False,
                              title=f"{model_name} Federated Confusion Matrix", outpath=os.path.join(OUTPUT_DIR, f"{model_name}_federated_cm.png"))
        plot_confusion_matrix(res['confusion_matrix'], class_names, normalize=True,
                              title=f"{model_name} Federated Confusion Matrix (Normalized)", outpath=os.path.join(OUTPUT_DIR, f"{model_name}_federated_cm_norm.png"))

        # Save predictions csv
        csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_federated_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "true_label", "pred_label"] + [f"prob_{c}" for c in class_names])
            for i, (t, p, prob) in enumerate(zip(res['labels'], res['preds'], res['probs'])):
                writer.writerow([i, class_names[t], class_names[p]] + list(map(float, prob)))

        print(f"Test Accuracy: {res['accuracy']:.4f}")
        print("Classification report:\n", res['classification_report'])

    # ---------------------------
    # Visualize some sample predictions for the best federated model (by test accuracy)
    # ---------------------------
    best_federated_model_name = max(overall_federated_results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest federated model on test set: {best_federated_model_name} (accuracy={overall_federated_results[best_federated_model_name]['accuracy']:.4f})")
    best_federated_model = trained_federated_models[best_federated_model_name]
    best_federated_model.to(DEVICE)

    # Show some test images with predictions
    def show_predictions(model, loader, classes, device, num_images=12):
        model.eval()
        images_shown = 0
        plt.figure(figsize=(14,10))
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                inputs = inputs.cpu()
                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    ax = plt.subplot(3, 4, images_shown+1)
                    imshow_tensor(inputs[i], mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                    true_lbl = classes[labels[i].item()]
                    pred_lbl = classes[preds[i].item()]
                    ax.set_title(f"T:{true_lbl}\\nP:{pred_lbl}\\n{probs[i].max().item():.2f}")
                    images_shown += 1
                if images_shown >= num_images:
                    break
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{best_federated_model_name}_federated_sample_predictions.png"))
        plt.show()

    show_predictions(best_federated_model, test_loader, class_names, DEVICE, num_images=12)

    # Save overall summary
    summary = {}
    for mname, res in overall_federated_results.items():
        summary[mname] = {"test_accuracy": float(res['accuracy'])}
    with open(os.path.join(OUTPUT_DIR, "summary_federated_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("All done. Federated results saved to:", OUTPUT_DIR)
