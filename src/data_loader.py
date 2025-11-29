# Data loading for Fashion-MNIST and CIFAR-10
import numpy as np
import os
import gzip
import pickle
import requests
import shutil
import tarfile
import struct

def download_fashion_mnist(data_dir='./data'):
    # Download Fashion-MNIST files
    os.makedirs(data_dir, exist_ok=True)
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
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

def download_cifar10(data_dir='./data'):
    # Download and extract CIFAR-10
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(extracted_dir):
        print("CIFAR-10 already downloaded and extracted")
        return
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if os.path.exists(tar_path):
        print("CIFAR-10 tar file already exists, extracting...")
    else:
        print("Downloading CIFAR-10 dataset (this may take a while)...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end='', flush=True)
        print(f"\nDownloaded CIFAR-10 to {tar_path}")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Download failed: {tar_path} does not exist")
        file_size = os.path.getsize(tar_path)
        if file_size == 0:
            raise ValueError(f"Download failed: {tar_path} is empty")
        print(f"File size: {file_size / (1024*1024):.1f} MB")
    print("Extracting CIFAR-10...")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Cannot extract: {tar_path} does not exist")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extracted CIFAR-10")
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Extraction failed: {extracted_dir} does not exist")
    os.remove(tar_path)
    print("Cleanup complete")

def load_fashion_mnist(data_dir='./data'):
    # Load Fashion-MNIST from IDX files
    def read_idx_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in {filename}")
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, num_rows, num_cols)
    def read_idx_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in {filename}")
            num_labels = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
    train_images = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels

def load_cifar10(data_dir='./data'):
    # Load CIFAR-10 from pickle files
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    def load_batch(filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return data, np.array(labels)
    train_batches = []
    train_labels_list = []
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
        data, labels = load_batch(batch_file)
        train_batches.append(data)
        train_labels_list.append(labels)
    train_images = np.vstack(train_batches)
    train_labels = np.concatenate(train_labels_list)
    test_file = os.path.join(cifar_dir, 'test_batch')
    test_images, test_labels = load_batch(test_file)
    return train_images, train_labels, test_images, test_labels

def preprocess_data(X, y, num_classes=10, flatten=True, normalize=True):
    # Preprocess: flatten, normalize, one-hot encode
    X_processed = X.astype(np.float32)
    if flatten:
        X_processed = X_processed.reshape(X_processed.shape[0], -1)
    if normalize:
        X_processed = X_processed / 255.0
    y_processed = np.eye(num_classes)[y]
    return X_processed, y_processed

def create_mini_batches(X, y, batch_size=32, shuffle=True):
    # Create mini-batches for training
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    mini_batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    return mini_batches

def train_val_split(X, y, val_split=0.2, random_seed=None):
    # Split data into train and val sets
    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    val_size = int(len(X) * val_split)
    X_val = X[:val_size]
    X_train = X[val_size:]
    y_val = y[:val_size]
    y_train = y[val_size:]
    return X_train, X_val, y_train, y_val

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def get_class_names(dataset):
    # Get class names for dataset
    if dataset.lower() == 'fashion_mnist':
        return FASHION_MNIST_CLASSES
    elif dataset.lower() == 'cifar10':
        return CIFAR10_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
