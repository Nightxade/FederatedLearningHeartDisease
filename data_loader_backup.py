from torch.utils.data import DataLoader
import torch
import pandas as pd
from CreateDatasetClass import CreateDataset

# read datasets, with first column set as the indices
diabetes_pd = pd.read_csv('with_med_specialty_dataset.csv', index_col=0)

# shuffle with_ms_pd
diabetes_pd = diabetes_pd.sample(frac=1).reset_index(drop=True)

# find 80:20 index separator
train_test_index = int(len(diabetes_pd.index) / 5 * 4)

# separate with_ms_pd into training and testing datasets
train_pd = diabetes_pd.copy()[:train_test_index:]
test_pd = diabetes_pd.copy()[train_test_index::]


# create transform function for Dataset class
# Pandas DataFrame --> Tensor
def transform(dataset):
    return torch.tensor(dataset)


# create CreateDataset classes
train_dataset = CreateDataset(train_pd, transform)
test_dataset = CreateDataset(test_pd, transform)

# create DataLoaders
train_data_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
