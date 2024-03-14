import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import h5py
import cv2
import glob
import os

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Dataset:
    """ get or create dataset object associated with filepath """
    def __init__(self, name, ood=False):
        # dataset
        self.X_CE = None
        self.y_CE = None
        if name == "mnist":
            self.X_train, self.y_train, self.X_test, self.y_test, self.classes, self.n_classes = self.load_mnist('./', ood)
        if name == "fmnist":
            self.X_train, self.y_train, self.X_test, self.y_test, self.classes, self.n_classes = self.load_fmnist('./', ood)
        if name == "moon":
            self.X_train, self.y_train, self.X_test, self.y_test, self.n_classes = self.load_moon(0.1)
        # dataloaders
        self.train_dl = None
        self.test_dl = None
        self.centroid_dl = None
        self.centroid_dl_iter = None

    def load_moon(self, noise):
        X_train, y_train = datasets.make_moons(n_samples=1500, noise=noise, random_state=0)
        X_test, y_test = datasets.make_moons(n_samples=200, noise=noise, random_state=1)
        return torch.tensor(X_train).float(), torch.tensor(y_train), torch.tensor(X_test).float(), torch.tensor(y_test), 2

    def load_mnist(self, root_dir, ood_fmnist=False):
        if ood_fmnist:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root=root_dir, download=True, train=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=root_dir, download=True, train=False, transform=transform)
        X_train = torch.stack([x for x,_ in train_dataset], 0)
        X_test = torch.stack([x for x,_ in test_dataset], 0)
        return X_train, train_dataset.targets, X_test, test_dataset.targets, train_dataset.classes, len(train_dataset.classes)
    
    def load_fmnist(self, root_dir, ood_mnist=False):
        if ood_mnist:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
        train_dataset = torchvision.datasets.FashionMNIST(root=root_dir, download=True, train=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=root_dir, download=True, train=False, transform=transform)
        X_train = torch.stack([x for x,_ in train_dataset], 0)
        X_test = torch.stack([x for x,_ in test_dataset], 0)
        return X_train, train_dataset.targets, X_test, test_dataset.targets, train_dataset.classes, len(train_dataset.classes)

    def get_example_from_class(self, class_num, seed=None):
        X, y = self.X_train, self.y_train
        class_examples = []
        class_idx = np.where(y == class_num)[0]
        idx = np.random.choice(class_idx, 1, replace=True)
        return X[idx]
    
    def set_train_dataloader(self, batch_size, shuffle=True, drop_last=True):
        torch.random.manual_seed(0)
        random.seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        set = TensorDataset(self.X_train, self.y_train)
        self.train_dl = DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=drop_last)
    
    def set_test_dataloader(self, batch_size, n_samples=1, shuffle=False):
        A = self.X_test
        y = self.y_test
        if n_samples > 1:
            _, c, iw, ih = A.shape
            A = torch.tile(A.unsqueeze(1), (1, n_samples, 1, 1, 1)).reshape(-1, c, iw, ih)
            y = torch.tile(y.unsqueeze(1), (1, n_samples)).reshape(-1)
        set = TensorDataset(A, y)
        self.test_dl = DataLoader(set, batch_size=batch_size, shuffle=shuffle)
    
    def get_prototype_dataloader(self, batch_size, n_samples, shuffle=False):
        A = self.X_test
        y = self.y_test
        _, c, iw, ih = A.shape
        A = torch.tile(A.unsqueeze(1), (1, n_samples, 1, 1, 1)).reshape(-1, c, iw, ih)
        y = torch.tile(y.unsqueeze(1), (1, n_samples)).reshape(-1)
        set = TensorDataset(A, y)
        return DataLoader(set, batch_size=batch_size, shuffle=shuffle)
    
    def set_centroid_dataloader(self, batch_size, shuffle=True):
        torch.random.manual_seed(0)
        random.seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        if self.X_CE is None:
            set = TensorDataset(self.X_train, self.y_train)
        else: 
            set = TensorDataset(self.X_CE, self.y_CE)
        self.centroid_dl = DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=True)
    
    def reset_centroid_dataloader_iter(self):
        if self.X_CE is not None:
            self.centroid_dl_iter = iter(self.centroid_dl)

    def get_class_examples(self, n_ex, seed=None):
        if self.X_CE is None:
            X, y = self.X_train, self.y_train
        else: 
            X, y = self.X_CE, self.y_CE
        class_examples = []
        class_idxs = [np.where(y == i)[0] for i in range(0, self.n_classes)]
        if seed is not None:
            np.random.seed(seed)
        for i in range(self.n_classes):
            idx = np.random.choice(class_idxs[i], n_ex, replace=True)
            class_examples.append(X[idx])
        return class_examples
    
    def get_test_by_idx(self, idx, n_samples):
        A = self.X_test[idx]
        y = self.y_test[idx]
        if n_samples > 1:
            A = A.unsqueeze(0).repeat(tuple([n_samples]) + tuple([1]*len(A.shape)))
        return A, y
