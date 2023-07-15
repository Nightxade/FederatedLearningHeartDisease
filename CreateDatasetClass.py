from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


# create Dataset class
class CreateDataset(Dataset):
    def __init__(self, dataset, target_column: str or int, transform=None):  # initialize the dataset
        self.dataset = dataset
        if type(target_column) is str:
            self.input = transform(dataset.drop(target_column, axis=1).values, torch.float)  # transform input
            # for i in range(self.input.shape[1]):
                # print('attribute', i, 'values', len(np.bincount(self.input[:, i])), np.bincount(self.input[:, i]))
            # self.input = transform(dataset.drop(target_column, axis=1).to_numpy().astype('float64'))  # transform input
            self.output = transform(dataset[target_column].values, torch.float)  # transform output\
        else:
            column = dataset.columns[target_column]
            self.input = transform(dataset.drop(column, axis=1).values, torch.float)
            self.output = transform(dataset[column].values, torch.float)

    def __len__(self):  # support the len operation, e.g. len(dataset)
        return len(self.dataset)

    def __getitem__(self, idx):  # support the indexing operation, e.g. dataset[i]
        data = self.input[idx], self.output[idx]
        return data

