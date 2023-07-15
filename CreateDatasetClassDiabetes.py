from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create Dataset class
class CreateDatasetDiabetes(Dataset):
    def __init__(self, dataset, target_column: str or int, transform=None):  # initialize the dataset
        self.dataset = dataset
        if type(target_column) is str:
            self.input = dataset.drop(target_column, axis=1).values  # transform input
            self.input = torch.tensor(self.input, dtype=torch.float).to(device)
            # for i in range(self.input.shape[1]):
               # print('attribute', i, 'values', len(np.bincount(self.input[:, i])), np.bincount(self.input[:, i]))
            # self.input = transform(dataset.drop(target_column, axis=1).to_numpy().astype('float64'))  # transform input
            self.output = dataset[target_column].values  # transform output\
            self.output = torch.tensor(self.output, dtype=torch.long).to(device)

        else:
            column = dataset.columns[target_column]
            self.input = transform(dataset.drop(column, axis=1).values)
            self.output = transform(dataset[column].values)

    def __len__(self):  # support the len operation, e.g. len(dataset)
        return len(self.dataset)

    def __getitem__(self, idx):  # support the indexing operation, e.g. dataset[i]
        data = self.input[idx], self.output[idx]
        return data


def transform(dataset):  # transforms dataset to a tensor
    # return torch.from_numpy(dataset)
    return torch.tensor(dataset)


data = pd.read_csv('processed_diabetes_dataset.csv')
dataset = CreateDatasetDiabetes(data, 'readmitted', transform)
