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
from cub_dataset import CUB

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Dataset:
    """ get or create dataset object associated with filepath """
    def __init__(self, name, ood=False):
        # dataset
        root_dir = '/users/hsit/scratch/'
        self.X_CE = None
        self.y_CE = None
        if name == "mnist":
            self.X_train, self.y_train, self.X_test, self.y_test, self.classes, self.n_classes = self.load_mnist(root_dir, ood)
        if name == "fmnist":
            self.X_train, self.y_train, self.X_test, self.y_test, self.classes, self.n_classes = self.load_fmnist(root_dir, ood)
        if name == "moon":
            self.X_train, self.y_train, self.X_test, self.y_test, self.n_classes = self.load_moon(0.1)
        if name == "cifar10":
            self.X_CE, self.y_CE, self.X_test, self.y_test, self.n_classes = self.load_cifar10(root_dir)
            self.X_train, self.y_train = self.load_cifar10_augment(root_dir)
        if name == "bird":
            self.X_CE, self.y_CE, self.X_test, self.n_classes = self.load_bird(root_dir)
            self.X_train, self.y_train = self.load_bird_augment(root_dir)
        
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
    
    def load_cifar10(self, root_dir):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=False, transform=transform)
        X_train = torch.stack([x for x,_ in train_dataset], 0)
        return X_train, train_dataset.targets, test_dataset, test_dataset.targets, len(train_dataset.classes)

    def load_cifar10_augment(self, root_dir):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=True, transform=transform)
        return train_dataset, train_dataset.targets
    
    def load_bird(self, root_dir):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        train_dataset = CUB(f'{root_dir}cub2002011/CUB_200_2011/', 'train', transform=transform)
        X_train = []
        y_train = []
        for x, y in train_dataset:
            y_train.append(y)
            X_train.append(x)
        y_train = torch.tensor(y_train)
        X_train = torch.stack(X_train, 0)
        test_dataset = CUB(f'{root_dir}cub2002011/CUB_200_2011/', 'test', transform=transform)
        return X_train, y_train, test_dataset, 200

    def load_bird_augment(self, root_dir):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        train_dataset = CUB(f'{root_dir}cub2002011/CUB_200_2011/', 'train', transform=transform)
        y_train = []
        for x, y in train_dataset:
            y_train.append(y)
        y_train = torch.tensor(y_train)
        return train_dataset, y_train
    
    def set_train_dataloader(self, batch_size, seed=0, drop_last=True):
        torch.random.manual_seed(seed)
        random.seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        try:
            set = TensorDataset(self.X_train, self.y_train)
        except:
            set = self.X_train
        self.train_dl = DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=drop_last)
    
    def set_test_dataloader(self, batch_size, shuffle=False):
        try:
            set = TensorDataset(self.X_test, self.y_test)
        except:
            set = self.X_test
        self.test_dl = DataLoader(set, batch_size=batch_size, shuffle=shuffle)
    
    def get_prototype_dataloader(self, batch_size, n_samples, shuffle=False):
        A = self.X_test
        y = self.y_test
        _, c, iw, ih = A.shape
        A = torch.tile(A.unsqueeze(1), (1, n_samples, 1, 1, 1)).reshape(-1, c, iw, ih)
        y = torch.tile(y.unsqueeze(1), (1, n_samples)).reshape(-1)
        set = TensorDataset(A, y)
        return DataLoader(set, batch_size=batch_size, shuffle=shuffle)
    
    def set_centroid_dataloader(self, batch_size, seed=0, drop_last=True):
        torch.random.manual_seed(seed)
        random.seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        if self.X_CE is None:
            try:
                set = TensorDataset(self.X_train, self.y_train)
            except:
                set = self.X_train
        else: 
            try:
                set = TensorDataset(self.X_CE, self.y_CE)
            except:
                set = self.X_CE
        self.centroid_dl = DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=drop_last)
    
    def reset_centroid_dataloader_iter(self):
        if self.X_CE is not None:
            self.centroid_dl_iter = iter(self.centroid_dl)

    def get_example_from_class(self, class_num, seed=None):
        class_idx = np.where(np.array(self.y_train) == class_num)[0]
        idx = np.random.choice(class_idx, 1, replace=True)[0]
        if self.X_CE is None:
            X = self.X_train[idx]
        else:
            X = self.X_CE[idx]
        return X

    def get_class_examples(self, n_ex, seed=None):
        class_examples = []
        # try:
        class_idxs = [np.where(np.array(self.y_train) == i)[0] for i in range(0, self.n_classes)]
        if seed is not None:
            np.random.seed(seed)
        for i in range(self.n_classes):
            idx = np.random.choice(class_idxs[i], n_ex, replace=True)
            try:
                class_examples.append(self.X_CE[idx])
            except:
                class_examples.append(self.X_train[idx])
        return class_examples
    
    def get_test_by_idx(self, idx, n_samples):
        try:
            A = self.X_test[idx]
            y = self.y_test[idx]
            if n_samples > 1:
                A = A.unsqueeze(0).repeat(tuple([n_samples]) + tuple([1]*len(A.shape)))
        except:
            X, y = self.test_dataset[idx]
            X = X.unsqueeze(0).tile(n_samples, 1, 1, 1)
        return A, y
