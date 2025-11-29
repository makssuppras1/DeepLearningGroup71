# Data loading and preprocessing utilities for Fashion-MNIST and CIFAR-10

import numpy as np
import os
from typing import Tuple, Optional
import gzip
import pickle
import requests
import shutil
import tarfile
import struct

def download_fashion_mnist(data_dir: str = './data') -> None:
    # Download Fashion-MNIST dataset files to data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz", # images
        "train-labels-idx1-ubyte.gz", # labels
        "t10k-images-idx3-ubyte.gz", # test images
        "t10k-labels-idx1-ubyte.gz" # test labels
    ]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists, skipping...")
            continue
        
        print(f"Downloading {filename}...")
        url = base_url + filename
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")

def download_cifar10(data_dir: str = './data') -> None:
    # Download and extract CIFAR-10 dataset to data_dir
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if already extracted
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(extracted_dir):
        print("CIFAR-10 already downloaded and extracted")
        return
    
    # Download the dataset (170MB)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    # Check if tar file already exists
    if os.path.exists(tar_path):
        print("CIFAR-10 tar file already exists, extracting...")
    else:
        print("Downloading CIFAR-10 dataset (this may take a while)...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Stream download in chunks to avoid memory issues
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192  # 8KB chunks
        
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end='', flush=True)
        
        print(f"\nDownloaded CIFAR-10 to {tar_path}")
        
        # Verify file exists and has content
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Download failed: {tar_path} does not exist")
        file_size = os.path.getsize(tar_path)
        if file_size == 0:
            raise ValueError(f"Download failed: {tar_path} is empty")
        print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    # Extract the dataset
    print("Extracting CIFAR-10...")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Cannot extract: {tar_path} does not exist")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extracted CIFAR-10")
    
    # Verify extraction succeeded
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Extraction failed: {extracted_dir} does not exist")
    
    # Remove the tar file to save space
    os.remove(tar_path)
    print("Cleanup complete")


def load_fashion_mnist(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load Fashion-MNIST dataset from IDX files
    # Returns: (X_train, y_train, X_test, y_test)
    # Images: (num_samples, 28, 28), Labels: (num_samples,), Values: 0-255 (uint8)
    
    def read_idx_images(filename):
        # Read IDX format image file
        with gzip.open(filename, 'rb') as f:
            # Read and verify magic number (2051 for images)
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in {filename}")
            
            # Read dimensions
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            
            # Read pixel data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, num_rows, num_cols)
    
    def read_idx_labels(filename):
        # Read IDX format label file
        with gzip.open(filename, 'rb') as f:
            # Read and verify magic number (2049 for labels)
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in {filename}")
            
            # Read number of labels
            num_labels = struct.unpack('>I', f.read(4))[0]
            
            # Read label data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
    
    # Load training data
    train_images = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    
    # Load test data
    test_images = read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    return train_images, train_labels, test_images, test_labels


def load_cifar10(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load CIFAR-10 dataset from pickle files
    # Returns: (X_train, y_train, X_test, y_test)
    # Images: (num_samples, 32, 32, 3) RGB, Labels: (num_samples,), Values: 0-255 (uint8)
    
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    def load_batch(filename):
        # Load a single CIFAR-10 batch file (pickle format)
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            # Keys are bytes: b'data', b'labels', b'batch_label', b'filenames'
            data = batch[b'data']      # Shape: (10000, 3072)
            labels = batch[b'labels']  # List of 10000 integers
            
            # Reshape from (10000, 3072) to (10000, 3, 32, 32)
            # Then transpose to (10000, 32, 32, 3) for standard RGB format
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            
            return data, np.array(labels)
    
    # Load all 5 training batches
    train_batches = []
    train_labels_list = []
    
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
        data, labels = load_batch(batch_file)
        train_batches.append(data)
        train_labels_list.append(labels)
    
    # Combine all training batches
    train_images = np.vstack(train_batches)  # Shape: (50000, 32, 32, 3)
    train_labels = np.concatenate(train_labels_list)  # Shape: (50000,)
    
    # Load test batch
    test_file = os.path.join(cifar_dir, 'test_batch')
    test_images, test_labels = load_batch(test_file)
    
    return train_images, train_labels, test_images, test_labels


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = 10,
    flatten: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Preprocess data: flatten images, normalize to [0,1], one-hot encode labels
    # Returns: (X_processed, y_processed) where y_processed is one-hot encoded
    X_processed = X.astype(np.float32)
    
    # Flatten images if needed (keeps batch dimension)
    if flatten:
        X_processed = X_processed.reshape(X_processed.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    if normalize:
        X_processed = X_processed / 255.0
    
    # One-hot encode the labels
    y_processed = np.eye(num_classes)[y]
    
    return X_processed, y_processed


def create_mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> list:
    # Create mini-batches for training
    # Returns: List of (X_batch, y_batch) tuples
    # Shuffle data before batching if requested
    
    # Shuffle the data if requested
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    # create the mini-batches
    mini_batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    return mini_batches

def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split data into training and validation sets
    # Returns: (X_train, X_val, y_train, y_val)
    # val_split: fraction of data for validation (e.g., 0.2 = 20%)
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split the data into training and validation sets
    val_size = int(len(X) * val_split)
    X_val = X[:val_size]
    X_train = X[val_size:]
    y_val = y[:val_size]
    y_train = y[val_size:]
    
    return X_train, X_val, y_train, y_val


# Dataset information
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_class_names(dataset: str) -> list:
    # Get class names for a dataset ('fashion_mnist' or 'cifar10')
    if dataset.lower() == 'fashion_mnist':
        return FASHION_MNIST_CLASSES
    elif dataset.lower() == 'cifar10':
        return CIFAR10_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

