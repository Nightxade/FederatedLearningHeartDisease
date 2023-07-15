from torch.utils.data import DataLoader
import torch
import pandas as pd
from CreateDatasetClass import CreateDataset

"""

# read datasets, with first column set as the indices
# with_ms_pd = pd.read_csv('with_med_specialty_dataset.csv', index_col=0)
# without_ms_pd = pd.read_csv('without_med_specialty_dataset.csv', index_col=0)
dataset_pd = pd.read_csv('processed_diabetes_dataset.csv', index_col=0)

# shuffle dataset
# with_ms_pd = with_ms_pd.sample(frac=1).reset_index(drop=True)
dataset_pd = dataset_pd.sample(frac=1).reset_index(drop=True)

# find 80:20 index separator
# train_test_index = int(len(with_ms_pd.index) / 5 * 4)
train_test_index = int(len(dataset_pd.index) / 5 * 4)

# separate with_ms_pd into training and testing datasets
train_pd = dataset_pd.copy()[:train_test_index:]
test_pd = dataset_pd.copy()[train_test_index::]

# evenly split training dataset
NUM_CLIENTS = 130
client_dataset_size = int(len(train_pd.index) / NUM_CLIENTS) # find size of datasets
training_pds = []
# iterate through every client but the last
for i in range(NUM_CLIENTS - 1):
    client_dataset = train_pd.copy()[i * client_dataset_size : (i + 1) * client_dataset_size :]
    training_pds.append(client_dataset)
# last client receives the rest
training_pds.append(train_pd.copy()[(NUM_CLIENTS - 1) * client_dataset_size ::])


# create transform function for Dataset class
# Pandas DataFrame --> Tensor
def transform(dataset):
    return torch.tensor(dataset)


# create CreateDataset classes
train_datasets = []
for i in range(len(training_pds)):
    train_datasets.append(CreateDataset(training_pds[i], transform))
train_entire_dataset = CreateDataset(with_ms_train_pd, transform)
test_dataset = CreateDataset(with_ms_test_pd, transform)
without_ms_dataset = CreateDataset(without_ms_pd, transform)

# create DataLoaders
train_data_loaders = []
for i in range(len(train_datasets)):
    train_data_loaders.append(DataLoader(train_datasets[i], batch_size=30, shuffle=True))
train_entire_dataset = DataLoader(train_entire_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
predict_data_loader = DataLoader(without_ms_dataset, batch_size=1000, shuffle=True)

"""