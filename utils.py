import torch
from torchvision import datasets
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import random 
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt 
import pandas as pd

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g 

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           generator,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True):
    """
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx, generator=generator)
    valid_sampler = SubsetRandomSampler(valid_idx, generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, generator=generator, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, generator=generator, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader, train_dataset, valid_idx)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader, dataset

def train_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
        test_noise_dict,
        w_bkup,
        lr_decay_gamma=0.1,
        lr_decay_milestones="8,9",
        regularizer=None,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]

    model.to(device)
    best_acc = -1
    
    grad_dict = {}
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if regularizer is not None:
                regularizer(model)  # for sparsity learning                
            
            with torch.no_grad():    
            
                for l, k in model.named_parameters():
                        k-=lr*k.grad.data
                        k.grad.zero_()
            
            with torch.no_grad():
                for x, y in model.named_modules():
                    if x in test_noise_dict:
                    # if x in noise_dict:
                        y.weight.data.view(-1)[test_noise_dict[x]] = w_bkup[x].to(device)

                        
            if i % 100 == 0:
                print(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(), 
                        lr
                        # optimizer.param_groups[0]["lr"]
                        ))

        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)

        print(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}".format
            (epoch, epochs, acc, val_loss)
        )

def eval(model, test_loader, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.zero_grad()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    loss = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)

    accuracy, loss = (correct / total).item(), (loss / total).item()

    print(f"Accuracy = {accuracy:.6f}, Loss = {loss:.6f}")

    return accuracy, loss

def eval_superclass(model, superclass_labels, test_loader, device=None):
    correct = 0
    total = 0
    predicted_superclass = []
    actual_superclass = []
    predicted_class = []
    actual_class = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_class.extend(predicted.cpu().numpy())
            actual_class.extend(labels.cpu().numpy())

            predicted_superclass.extend([superclass_labels[label] for label in predicted.cpu().numpy()])
            actual_superclass.extend([superclass_labels[label] for label in labels.cpu().numpy()])
            
        superclass_accuracy = np.mean(np.array(predicted_superclass) == np.array(actual_superclass))
        
    return superclass_accuracy

def compute_class_accuracies(model, dataloader, device, num_classes=100):
    difficulties = []
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            difficulties.extend((predicted != labels).float().tolist())
            for label, pred in zip(labels, predicted):
                if label == pred:
                    correct[label] += 1
                total[label] += 1
    
    accuracies = correct / total
    return accuracies.cpu().numpy(), difficulties

def create_imbalanced_sampler(dataset, indices, class_accuracies, num_samples):
    labels = np.array(dataset.targets)[indices]
    # print(labels, len(labels))
    class_weights = 1 / (class_accuracies + 1e-5)  # Add small epsilon to avoid division by zero
    sample_weights = class_weights[labels]
    # print(len(sample_weights))
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=False)

def weight_by_subclass(subclass_acc):
    """Most obvious weighting scheme, where each subclass is weighted in 
    proportion to its error"""
    subclass_errors = 1 - subclass_acc
    # It is sufficient to use subclass_errors as the weights if the weights
    # don't have to sum to one. Otherwise, we normalize explicitly:
    sum_of_errors = np.sum(subclass_errors)
    subclass_weights = subclass_errors / sum_of_errors
    
    return subclass_weights

def weight_by_superclass(subclass_acc, homogeneity=1.0):
    """As weight_by_subclass, but each subclass weight is blended with the  
       weight of its superclass. Homogeneity of 0 makes this function equivalent
       to weight_by_subclass, while homogeneity of 1 assigns each subclass the
       average weight of its superclass."""
    
    subclass_weights = weight_by_subclass(subclass_acc)
    print(subclass_weights, len(subclass_weights))
    superclass_weights = np.average(np.split(subclass_weights, 20), axis=-1)
    print(superclass_weights, len(superclass_weights))
    blended_weights = (1.0 - homogeneity) * subclass_weights + \
                      homogeneity * np.repeat(superclass_weights, 5)
    return blended_weights

def class_mapping():
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    return {class_name: index for index, class_name in enumerate(class_names)}

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(3, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            nn.MaxPool2d(2, 2),
            self.conv_block(64, 128),
            self.conv_block(128, 64),
            nn.MaxPool2d(2, 2),
            self.conv_block(64, 32),
            self.conv_block(32, 16),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self.conv_block(8, 16),
            self.conv_block(16, 32),
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.conv_block(128, 64),
            self.conv_block(64, 32),
            self.conv_block(32, 16),
            nn.Conv2d(16, 3, 3, padding=1)
            # nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
from typing import Tuple, Optional
import matplotlib.gridspec as gridspec

def compute_calibration_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_bins: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute calibration metrics for reliability diagram.
    """
    model.eval()
    confidences_list = []
    accuracies_list = []
    entropy_list = []
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Get model predictions and confidences
            logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            # Convert to numpy for easier handling
            confidences_list.extend(confidences.cpu().numpy())
            accuracies_list.extend((predictions == targets).cpu().numpy())
    
    confidences = np.array(confidences_list)
    accuracies = np.array(accuracies_list)
    
    # Create bins and compute metrics
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies_in_bins = []
    confidences_in_bins = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        count = np.sum(in_bin)
        counts.append(count)
        
        if count > 0:
            accuracies_in_bins.append(np.mean(accuracies[in_bin]))
            confidences_in_bins.append(np.mean(confidences[in_bin]))
        else:
            accuracies_in_bins.append(0)
            confidences_in_bins.append(bin_lower + (bin_upper - bin_lower) / 2)
    
    # Calculate Expected Calibration Error (ECE)
    ece = np.sum(np.abs(np.array(accuracies_in_bins) - np.array(confidences_in_bins)) * 
                 (np.array(counts) / np.sum(counts)))
    
    return (np.array(accuracies_in_bins), 
            np.array(confidences_in_bins), 
            np.array(counts), 
            ece)

def plot_clear_reliability_diagram(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    counts: np.ndarray,
    ece: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a clearer reliability diagram with separate subplots for calibration and counts.
    """
    # Create figure with GridSpec for better control over subplot sizes
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Top subplot: Reliability Diagram
    ax1 = plt.subplot(gs[0])
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.plot(confidences, accuracies, 'ro-', label='Model calibration', linewidth=2)
    
    # Fill between perfect calibration and model calibration
    ax1.fill_between(confidences, 
                     confidences,  # perfect calibration line
                     accuracies,   # actual calibration line
                     alpha=0.2,
                     color='red',
                     label='Calibration gap')
    
    # Customize top subplot
    ax1.set_xlabel('Confidence (Predicted Probability)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title or f'Reliability Diagram (ECE: {ece:.3f})')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Bottom subplot: Prediction Distribution
    ax2 = plt.subplot(gs[1])
    ax2.bar(confidences, counts, width=0.1, color='blue', alpha=0.6,
            label='Number of predictions')
    
    # Customize bottom subplot
    ax2.set_xlabel('Confidence (Predicted Probability)')
    ax2.set_ylabel('Count of Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def entropy(probs):
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1. - epsilon)
    return -np.sum(probs * np.log(probs))


def analyze_model_calibration_clear(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Analyze and visualize model calibration with clearer visualization.
    Returns the Expected Calibration Error (ECE).
    """
    # Compute calibration metrics
    accuracies, confidences, counts, ece = compute_calibration_metrics(
        model, test_loader, num_bins, device
    )
    
    # Plot reliability diagram
    plot_clear_reliability_diagram(accuracies, confidences, counts, ece, title, save_path)
    
    return ece

# Normalize the images
def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def analyze_test_set_uncertainty(model, test_loader, device, top_k=10):
    """
    Analyze the uncertainty of model predictions on a test set.
    Assumes the model outputs logits.
    
    :param model: Trained PyTorch model
    :param test_loader: DataLoader for the test set
    :param device: Device to run the model on ('cuda' or 'cpu')
    :param top_k: Number of most uncertain predictions to return
    :return: List of (image_index, entropy, image_tensor, true_class, predicted_class) tuples
    """
    model.eval()
    results = []
    entrop = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get predicted classes
            _, predicted = torch.max(logits, 1)
            
            # Calculate entropy for each prediction in the batch
            for j, (prob, label, pred) in enumerate(zip(probs, labels, predicted)):
                img_entropy = entropy(prob.cpu().numpy())
                results.append((
                    i * test_loader.batch_size + j,  # image index
                    img_entropy,                     # entropy
                    inputs[j].cpu(),                 # image tensor
                    label.item(),                    # true class
                    pred.item()                      # predicted class
                ))
                entrop.append(img_entropy)
    
    # Sort results by entropy in descending order and return top_k
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k], entrop

from typing import Dict, Any
def save_array_results(results: Dict[str, np.ndarray], seed: int, base_filename: str = 'results'):
    """
    Save array results to separate CSV files, one for each metric.
    Each seed's results become a new column in each CSV.
    
    Args:
        results: Dictionary containing array results
        seed: Random seed used for this run
        base_filename: Base name for CSV files (will append metric name)
    """
    for metric, values in results.items():
        filename = f"3000_{base_filename}_{metric}.csv"
        
        if not os.path.exists(filename):
            # First run - create new file
            df = pd.DataFrame({str(seed): values})
            df.to_csv(filename, index=False)
            print(f"Created new file for {metric}: {filename}")
        else:
            # Read existing file
            df = pd.read_csv(filename)
            # Add new column with current seed's results
            df[str(seed)] = values
            # Save updated dataframe
            df.to_csv(filename, index=False)
            print(f"Added seed {seed} results to {filename}")
