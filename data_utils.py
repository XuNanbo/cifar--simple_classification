"""数据加载与 Dataset 封装"""
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset

__all__ = ['load_cifar10', 'CIFAR10Dataset']

def _unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(root):
    train_imgs, train_lbls = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(root, f'data_batch_{i}'))
        train_imgs.append(d[b'data'])
        train_lbls.extend(d[b'labels'])
    train_imgs = np.vstack(train_imgs)
    train_lbls = np.array(train_lbls, dtype=np.int64)

    test_dict = _unpickle(os.path.join(root, 'test_batch'))
    test_imgs = test_dict[b'data']
    test_lbls = np.array(test_dict[b'labels'], dtype=np.int64)
    return train_imgs, train_lbls, test_imgs, test_lbls

class CIFAR10Dataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].reshape(3,32,32)
        img = torch.tensor(img, dtype=torch.float32) / 255.
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
