import os
import pickle
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torchvision
from scipy.io import loadmat


def load_mnist(path="datasets/MNIST", valid_size=10000, seed=42):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Load Dataset
    os.makedirs(path, exist_ok=True)

    train = torchvision.datasets.MNIST(root=path, download=True, train=True)
    test = torchvision.datasets.MNIST(root=path, download=True, train=False)

    train_size = 60000 - valid_size
    X_train = train.data.numpy()[:train_size].reshape(train_size, 784) / 255
    X_valid = train.data.numpy()[train_size:].reshape(valid_size, 784) / 255
    X_test = test.data.numpy().reshape(10000, 784) / 255

    # Y_train = train.train_labels.numpy()[:train_size]
    # Y_valid = train.train_labels.numpy()[train_size:]
    # Y_test = test.test_labels.numpy()

    # return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    return X_train, X_valid, X_test


def load_omniglot(base_dir="datasets/OMNIGLOT", valid_size=1345, seed=42):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    file_name = "chardata.mat"
    file_path = Path(base_dir).joinpath(file_name)

    if not file_path.exists():
        # Download original data
        print("Downloading Datasets")
        os.makedirs(base_dir, exist_ok=True)

        url = "https://raw.githubusercontent.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat"
        urllib.request.urlretrieve(url, str(file_path))

    # Load Dataset
    omniglot = loadmat(str(file_path))
    train_data = np.hstack([omniglot['data'].T, omniglot['target'].T])
    np.random.shuffle(train_data)

    X_valid = train_data[:valid_size, :784]
    X_train = train_data[valid_size:, :784]
    X_test = omniglot['testdata'].T

    # Y_valid = train_data[:valid_size, 784:]
    # Y_train = train_data[valid_size:, 784:]
    # Y_test = omniglot['testtarget'].T

    # return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    return X_train, X_valid, X_test


def load_histopathology(base_dir="datasets/Histopathology", seed=42):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    tar_name = "histopathology.pkl.tar.gz"
    file_name = "histopathology.pkl"

    tar_path = Path(base_dir).joinpath(tar_name)
    file_path = Path(base_dir).joinpath(file_name)

    if not tar_path.exists():
        # Download original data
        print("Downloading Datasets")
        os.makedirs(base_dir, exist_ok=True)

        url = ("https://raw.githubusercontent.com/jmtomczak/vae_householder_flow/"
               "master/datasets/histopathologyGray/histopathology.pkl.tar.gz")
        urllib.request.urlretrieve(url, str(tar_path))

    # Extract tarball
    tar = tarfile.open(str(tar_path))
    tar.extractall(path=base_dir)
    tar.close()

    # Load Dataset
    with open(str(file_path), mode="rb") as f:
        data = pickle.load(f, encoding="latin1")

    X_train = np.asarray(data['training']).reshape(-1, 28 * 28)
    X_valid = np.asarray(data['validation']).reshape(-1, 28 * 28)
    X_test = np.asarray(data['test']).reshape(-1, 28 * 28)

    X_train = np.clip(X_train, 1 / 512, 1 - 1 / 512)
    X_valid = np.clip(X_valid, 1 / 512, 1 - 1 / 512)
    X_test = np.clip(X_test, 1 / 512, 1 - 1 / 512)

    return X_train, X_valid, X_test


def load_freyfaces(base_dir="datasets/FreyFaces", train_size=1565, valid_size=200, test_size=200, seed=42):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    file_name = "freyfaces.pkl"
    file_path = Path(base_dir).joinpath(file_name)

    if not file_path.exists():
        # Download original data
        print("Downloading Datasets")
        os.makedirs(base_dir, exist_ok=True)

        url = "https://raw.githubusercontent.com/jmtomczak/vae_vampprior/master/datasets/Freyfaces/freyfaces.pkl"
        urllib.request.urlretrieve(url, str(file_path))

    # Load Dataset
    with open(str(file_path), mode="rb") as f:
        data = pickle.load(f, encoding="latin1")

    data = data[0] / 256
    np.random.shuffle(data)

    X_train = data[0:train_size]
    X_valid = data[train_size:(train_size + valid_size)]
    X_test = data[(train_size + valid_size):(train_size + valid_size + test_size)]

    return X_train, X_valid, X_test


def load_one_hot(seed=42):
    # Seed
    random.seed(seed)
    np.random.seed(seed)

    Y_train = np.hstack([
        np.array([0] * 250),
        np.array([1] * 250),
        np.array([2] * 250),
        np.array([3] * 250)
    ])
    np.random.shuffle(Y_train)
    X_train = np.identity(4)[Y_train]

    Y_valid = np.hstack([
        np.array([0] * 25),
        np.array([1] * 25),
        np.array([2] * 25),
        np.array([3] * 25)
    ])
    np.random.shuffle(Y_valid)
    X_valid = np.identity(4)[Y_valid]

    Y_test = np.hstack([
        np.array([0] * 250),
        np.array([1] * 250),
        np.array([2] * 250),
        np.array([3] * 250)
    ])
    np.random.shuffle(Y_test)
    X_test = np.identity(4)[Y_test]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_dataset(key):
    loader = {
        "MNIST": load_mnist,
        "OMNIGLOT": load_omniglot,
        "Histopathology": load_histopathology,
        "FreyFaces": load_freyfaces,
        "OneHot": load_one_hot
    }
    return loader[key]()
