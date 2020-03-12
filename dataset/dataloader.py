# Returning the train set, validation set, and test set of the MNIST, CIFAR10, and CIFAR100 dataset
import torch
from torchvision import datasets, transforms
from torch import utils
import numpy as np

def mnist_loader(args, train_batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_set =datasets.MNIST('./dataset',
                               train=True,
                               download=True,
                               transform=transform)
    test_set =datasets.MNIST('./dataset',
                             train=False,
                             download=True,
                             transform=transform)
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(args.split_fraction * num_train))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    valid_indices, train_indices = indices[:split], indices[split:]
    train_sampler= utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = utils.data.DataLoader(dataset=train_set,
                                         batch_size=train_batch_size,
                                         sampler=train_sampler,
                                         **kwargs)

    val_loader = utils.data.DataLoader(dataset=train_set,
                                     batch_size=train_batch_size,
                                     sampler=val_sampler,
                                     **kwargs)

    test_loader = utils.data.DataLoader(dataset=test_set,
                                        batch_size=test_batch_size,
                                        shuffle=False,
                                        **kwargs)
    return train_loader, val_loader, test_loader

def cifar10_loader(args, train_batch_size, test_batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./dataset',
                                 train=True,
                                 download=True,
                                 transform=transform)
    test_set = datasets.CIFAR10(root='./dataset',
                                train=False,
                                download=True,
                                transform=transform)
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(args.split_fraction * num_train))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    valid_indices, train_indices = indices[:split], indices[split:]
    train_sampler= utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = utils.data.DataLoader(dataset=train_set,
                                         batch_size=train_batch_size,
                                         sampler=train_sampler,
                                         **kwargs)

    val_loader = utils.data.DataLoader(dataset=train_set,
                                     batch_size=train_batch_size,
                                     sampler=val_sampler,
                                     **kwargs)

    test_loader = utils.data.DataLoader(dataset=test_set,
                                        batch_size=test_batch_size,
                                        shuffle=False,
                                        **kwargs)
    return train_loader, val_loader, test_loader


def cifar100_loader(args, train_batch_size, test_batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])

    train_set = datasets.CIFAR100(root='./dataset',
                                              train=True,
                                              download=True,
                                              transform=train_transform,
                                              )
    test_set = datasets.CIFAR100(root='./dataset',
                                             train=False,
                                             download=True,
                                             transform=test_transform)
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(args.split_fraction * num_train))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    valid_indices, train_indices = indices[:split], indices[split:]
    train_sampler= utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = utils.data.DataLoader(dataset=train_set,
                                         batch_size=train_batch_size,
                                         sampler=train_sampler,
                                         **kwargs)

    val_loader = utils.data.DataLoader(dataset=train_set,
                                     batch_size=train_batch_size,
                                     sampler=val_sampler,
                                     **kwargs)

    test_loader = utils.data.DataLoader(dataset=test_set,
                                        batch_size=test_batch_size,
                                        shuffle=False,
                                        **kwargs)
    return train_loader, val_loader, test_loader
