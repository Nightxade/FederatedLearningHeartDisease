import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd

from CreateDatasetClass import CreateDataset
from CreateDatasetClassDiabetes import CreateDatasetDiabetes


# class for loading and managing dataset
class DataProvider:
    def __init__(self, dataset_name, num_clients, batch_size) -> None:
        self.batch_size = batch_size
        self.train_data, self.test_data = self.split_train_test_data(dataset_name)

    @staticmethod
    def transform(dataset, data_type):  # transforms dataset to a tensor
        # return torch.from_numpy(dataset)
        return torch.tensor(dataset, dtype=data_type)

    def split_train_test_data(self, dataset_name):
        if dataset_name == 'heart_disease':
            dataset_pd = pd.read_csv('heart_disease_cleveland.csv')  # read dataset
            dataset_pd = dataset_pd.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data

            train_test_index = int(len(dataset_pd.index) / 5 * 4)  # get index for 80:20 split
            train_pd = dataset_pd.copy()[:train_test_index:]
            test_pd = dataset_pd.copy()[train_test_index::]

            # create Dataset classes
            train_data = CreateDataset(train_pd, 13, self.transform)
            test_data = CreateDataset(test_pd, 13, self.transform)
        elif dataset_name == 'readmission':
            dataset_pd = pd.read_csv('processed_diabetes_dataset.csv')
            dataset_pd = dataset_pd.sample(frac=0.1, random_state=50).reset_index(drop=True)  # shuffle data

            dataset_pd = dataset_pd.drop(dataset_pd.columns[[18, 19, 20, 22, 25, 28, 29, 30, 31, 33, 34, 35, 36, 37]], axis=1)

            train_test_index = int(len(dataset_pd.index) * 0.9)  # get index for 80:20 split
            train_pd = dataset_pd.copy()[:train_test_index:]
            test_pd = dataset_pd.copy()[train_test_index::]

            # create Dataset classes
            train_data = CreateDatasetDiabetes(train_pd, 'readmitted', self.transform)
            test_data = CreateDatasetDiabetes(test_pd, 'readmitted', self.transform)
        else:
            assert dataset_name == 'heart_disease', 'Only the heart disease dataset is currently supported'

        return train_data, test_data

    def split_data_to_clients(self, dataset, num_clients):
        # Determine size of each split
        num_total_samples = len(dataset)
        num_samples_per_client = num_total_samples // num_clients

        # Create a list to store data for each client
        client_data_list = []

        # Split the dataset and assign to clients
        indices = list(range(num_total_samples))
        np.random.shuffle(indices)  # shuffle indices to randomize order

        # Split the dataset and assign to sub_dataset
        for i in range(num_clients):
            subset_indices = indices[i * num_samples_per_client: (
                                                                             i + 1) * num_samples_per_client]  # select subset of random indices
            client_data = DataLoader(Subset(dataset, subset_indices), batch_size=self.batch_size,
                                     shuffle=True)  # initialize DataLoader with the subset
            client_data_list.append(client_data)  # append current client DataLoader to list of DataLoaders

        return client_data_list
