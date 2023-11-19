import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


def get_csv_file(split):
    if split == 'train':
        csv_file = 'train_370.csv'
        print('Training with: ', csv_file)
    elif split == 'val':
        csv_file = 'val_370.csv'
        print('Validation: ', csv_file)
    elif split == 'test':
        csv_file = 'test_370.csv'
        print('Testing: ', csv_file)
    elif split == 'additional_test':
        csv_file = 'additional_test.csv'
        print('Additional testing: ', csv_file)
    return csv_file


class T1COR_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1COR_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t1COR_path = self.df['before_T1_COR_W']
        self.after_t1COR_path = self.df['after_T1_COR_W']
        self.malignant = self.df['PCR']


        self.malignant = torch.LongTensor(self.malignant)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1COR_pth = self.before_t1COR_path[index]
        after_t1COR_pth = self.after_t1COR_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1COR_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t1COR_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)
        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t1COR_path)


class T1COR_concate_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with CORresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1COR_concate_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t1COR_path = self.df['before_T1_COR_W']
        self.after_t1COR_path = self.df['after_T1_COR_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1COR_pth = self.before_t1COR_path[index]
        after_t1COR_pth = self.after_t1COR_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1COR_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t1COR_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        data = np.concatenate([before_data, after_data])
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t1COR_path)


class T1COR_before_Dataset(Dataset):
    def __init__(self, split, root, before_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with CORresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1COR_before_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.subject = self.df['name']
        self.before_t1COR_path = self.df['before_T1_COR_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1COR_pth = self.before_t1COR_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1COR_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)

        data = before_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t1COR_path)
    
class T1COR_after_Dataset(Dataset):
    def __init__(self, split, root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with CORresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1COR_after_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.after_root = after_root
        self.subject = self.df['name']
        self.after_t1COR_path = self.df['after_T1_COR_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        after_t1COR_pth = self.after_t1COR_path[index]
        after_data = np.load(os.path.join(self.after_root, after_t1COR_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            after_data = self.transform(after_data)

        data = after_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.after_t1COR_path)


class T1TRA_before_Dataset(Dataset):
    def __init__(self, split, root, before_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1TRA_before_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.subject = self.df['name']
        self.before_t1TRA_path = self.df['before_T1_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1TRA_pth = self.before_t1TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)

        data = before_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t1TRA_path)

class T1TRA_after_Dataset(Dataset):
    def __init__(self, split, root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1TRA_after_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.after_root = after_root
        self.subject = self.df['name']
        self.after_t1TRA_path = self.df['after_T1_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        after_t1TRA_pth = self.after_t1TRA_path[index]
        after_data = np.load(os.path.join(self.after_root, after_t1TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            after_data = self.transform(after_data)

        data = after_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.after_t1TRA_path)
    
class T2TRA_before_Dataset(Dataset):
    def __init__(self, split, root, before_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2TRA_before_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.subject = self.df['name']
        self.before_t2TRA_path = self.df['before_T2_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t2TRA_pth = self.before_t2TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t2TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)

        data = before_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t2TRA_path)

class T2TRA_after_Dataset(Dataset):
    def __init__(self, split, root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2TRA_after_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.after_root = after_root
        self.subject = self.df['name']
        self.after_t2TRA_path = self.df['after_T2_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        after_t2TRA_pth = self.after_t2TRA_path[index]
        after_data = np.load(os.path.join(self.after_root, after_t2TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            after_data = self.transform(after_data)

        data = after_data
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.after_t2TRA_path)
    
class T1COR_zhongliu_Dataset(Dataset):
    def __init__(self, root, csv_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1COR_zhongliu_Dataset, self).__init__()
        self.df = pd.read_csv(csv_root)
        self.root = root
        self.subject = self.df['num']
        self.before_t1cor_path = self.df['before_T1_COR']
        self.after_t1cor_path = self.df['after_T1_COR']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1cor_pth = self.before_t1cor_path[index]
        after_t1cor_pth = self.after_t1cor_path[index]
        before_data = np.load(os.path.join(self.root, before_t1cor_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.root, after_t1cor_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t1cor_path)


class T1TRA_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1TRA_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t1TRA_path = self.df['before_T1_TRA_W']
        self.after_t1TRA_path = self.df['after_T1_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1TRA_pth = self.before_t1TRA_path[index]
        after_t1TRA_pth = self.after_t1TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t1TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)
        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t1TRA_path)


class T1TRA_concate_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1TRA_concate_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t1TRA_path = self.df['before_T1_TRA_W']
        self.after_t1TRA_path = self.df['after_T1_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1TRA_pth = self.before_t1TRA_path[index]
        after_t1TRA_pth = self.after_t1TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t1TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t1TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        data = np.concatenate([before_data, after_data])
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t1TRA_path)


class T1TRA_zhongliu_Dataset(Dataset):
    def __init__(self, root, csv_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T1TRA_zhongliu_Dataset, self).__init__()
        self.df = pd.read_csv(csv_root)
        self.root = root
        self.subject = self.df['num']
        self.before_t1TRA_path = self.df['before_T1_TRA']
        self.after_t1TRA_path = self.df['after_T1_TRA']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t1TRA_pth = self.before_t1TRA_path[index]
        after_t1TRA_pth = self.after_t1TRA_path[index]
        before_data = np.load(os.path.join(self.root, before_t1TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.root, after_t1TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t1TRA_path)


class T2TRA_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2TRA_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t2TRA_path = self.df['before_T2_TRA_W']
        self.after_t2TRA_path = self.df['after_T2_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t2TRA_pth = self.before_t2TRA_path[index]
        after_t2TRA_pth = self.after_t2TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t2TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t2TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)
        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t2TRA_path)


class T2TRA_concate_Dataset(Dataset):
    def __init__(self, split, root, before_root, after_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2TRA_concate_Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        self.before_root = before_root
        self.after_root = after_root
        self.subject = self.df['name']
        self.before_t2TRA_path = self.df['before_T2_TRA_W']
        self.after_t2TRA_path = self.df['after_T2_TRA_W']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t2TRA_pth = self.before_t2TRA_path[index]
        after_t2TRA_pth = self.after_t2TRA_path[index]
        before_data = np.load(os.path.join(self.before_root, before_t2TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.after_root, after_t2TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        data = np.concatenate([before_data, after_data])
        data = data.astype(float)
        return index, data, label

    def __len__(self):
        return len(self.before_t2TRA_path)


class T2TRA_zhongliu_Dataset(Dataset):
    def __init__(self, root, csv_root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with TRAresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2TRA_zhongliu_Dataset, self).__init__()
        self.df = pd.read_csv(csv_root)
        self.root = root
        self.subject = self.df['num']
        self.before_t2TRA_path = self.df['before_T2_TRA']
        self.after_t2TRA_path = self.df['after_T2_TRA']
        self.malignant = self.df['PCR']
        # self.birads = self.df['BIRADS']

        self.malignant = torch.LongTensor(self.malignant)
        # self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        before_t2TRA_pth = self.before_t2TRA_path[index]
        after_t2TRA_pth = self.after_t2TRA_path[index]
        before_data = np.load(os.path.join(self.root, before_t2TRA_pth))[np.newaxis, :]
        after_data = np.load(os.path.join(self.root, after_t2TRA_pth))[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            before_data = self.transform(before_data)
            after_data = self.transform(after_data)

        before_data = before_data.astype(float)
        after_data = after_data.astype(float)
        return index, before_data, after_data, label

    def __len__(self):
        return len(self.before_t2TRA_path)