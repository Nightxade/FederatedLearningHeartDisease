import math

import torch
from torch import nn
from NeuralNetworkClass import NeuralNetwork
from torch.utils.data import DataLoader
import pandas as pd
from CreateDatasetClass import CreateDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)

#######################

# CONSTANTS
NUM_INPUT = 13
NUM_OUTPUT = 1

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# set neural network parameters
num_layers = 2
layer_sizes = [NUM_INPUT, 32, NUM_OUTPUT]
activation_functions = [nn.LeakyReLU(inplace=True), nn.Sigmoid()]
learning_rate = 1e-3
loss_function = nn.BCELoss()

# create NN
net = NeuralNetwork(num_layers, layer_sizes, activation_functions)
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

# create model
model = net


# print(model)


# train function
def train(model, device, train_data_loader, loss_function, optim):
    model.train()  # set model to training mode

    running_loss = 0
    total = 0
    correct = 0

    # iterate through batches
    for batch_index, data in enumerate(train_data_loader):
        features, labels = data  # separate data into features and label
        # features, labels = features.to(device), labels.to(device)  # move to device
        # print(features)
        # features = features.float()  # BUG FIX -- convert features tensor data type to float
        # print(features)
        output = model(features.float())  # get output / predictions
        # print(output.reshape(len(output)), labels)
        loss = loss_function(output.reshape(len(output)), labels.float().reshape(len(labels)))  # compute output loss # BUG FIX -- convert labels data type to long
        optim.zero_grad()  # zero gradients for each batch
        loss.backward()  # backward propagation -- compute output gradient
        optim.step()  # adjust learning weights for optimizer
        pred = torch.round(output)  # get index of max-log probability
        # print(pred.reshape(len(pred)), labels)
        # _, pred = torch.max(output.data, 1)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        # print(batch_index, " ", loss)

        # print(output.shape, labels.shape)
        # print(output.dtype, labels.long().dtype)
        # print(output, pred.reshape(pred.shape[0]), labels)

        running_loss += loss.item()
        total += labels.size(0)
        # print(output)
        # print(pred.reshape(len(pred)))
        # print(labels)
        # print(loss)
        # for layer in model.linear_stack:
        #     print(layer.weight.grad[0])
        #     print(layer.weight)
        #     break
        # if batch_index == 5: break
    # print(len(train_data_loader.dataset), len(train_data_loader))
    # print(f"Training Accuracy: {correct} / {len(train_data_loader.dataset)}, i.e. {100 * correct / len(train_data_loader.dataset)}")
    # print(running_loss, len(train_data_loader))
    return running_loss / len(train_data_loader)


# test function
def test(model, device, test_data_loader, loss_function):
    model.eval()  # set model to evaluation mode

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (features, labels) in enumerate(test_data_loader):
            # features, label = features.to(device), label.to(device)  # move to device
            # features = features.float()  # BUG FIX -- convert features tensor data type to float # BUG FIX -- convert labels data type to long
            output = model(features.float())  # compute output
            test_loss += loss_function(output.reshape(len(output)), labels.reshape(len(labels)))  # add batch loss to total
            pred = torch.round(output)  # get index of max-log probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            # if batch_index == 5: break

            # print(pred.reshape(len(pred)), labels)

    test_loss /= len(test_data_loader)  # compute total test loss

    # output test results
    # print(
    #     f'Test Set:\nAverage Loss: {test_loss}\nAccuracy = {correct}/{total} i.e. {100 * correct / total}\n')

    return 100 * correct / total, test_loss


# iterate through epochs
# for dataset in range(len(train_data_loaders)):
#     average_loss = 0
#     for epoch in range(1, NUM_EPOCHS + 1):
#         loss = train(model, device, train_data_loaders[dataset], epoch)  # train model and return loss
#         average_loss += loss
#         # print(f'\tEpoch {epoch} Loss: {loss}')
#     average_loss /= NUM_EPOCHS
#     print(f'Dataset {dataset + 1} Loss: {average_loss}')
#     test(model, device, test_data_loader)


def transform(dataset, data_type):  # transforms dataset to a tensor
    # return torch.from_numpy(dataset)
    return torch.tensor(dataset, dtype=data_type)


dataset_pd = pd.read_csv('heart_disease_cleveland.csv')
dataset_pd = dataset_pd.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data
dataset_pd.astype(float)
train_test_index = int(len(dataset_pd.index) / 5 * 4)  # get index for 80:20 split
train_pd = dataset_pd.copy()[:train_test_index:]
test_pd = dataset_pd.copy()[train_test_index::]

train_data = CreateDataset(train_pd, 13, transform)
test_data = CreateDataset(test_pd, 13, transform)

train_data = DataLoader(train_data, batch_size=16, shuffle=True)
test_data = DataLoader(test_data, batch_size=16, shuffle=False)


# def equal_var_init(model):
#     for name, param in model.named_parameters():
#         if name.endswith(".bias"):
#             param.data.fill_(0)
#         elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
#             param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
#         else:
#             param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


# equal_var_init(model)

print(model)

accuracy_over_epochs = []
loss_over_epochs = []
epochs_recorded = []
NUM_EPOCHS = 1000
for epoch in range(1, NUM_EPOCHS + 1):
    loss = train(model, device, train_data, loss_function, optim)
    test_accuracy, test_loss = test(model, device, test_data, loss_function)

    if epoch % 100 == 0:
        print(f'\tEpoch {epoch} Training Loss: {loss}')
        print(f'Test Set:\nLoss: {test_loss}\nAccuracy = {test_accuracy}\n')

    accuracy_over_epochs.append(test_accuracy)
    loss_over_epochs.append(test_loss)
    epochs_recorded.append(epoch)
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')
# average_loss /= NUM_EPOCHS
# print(f'Loss: {average_loss}')

# Plot Graphs

plt.style.use('Solarize_Light2')  # set plot style

# accuracy graph
plt.ylim(0, 100)  # set range
plt.plot(epochs_recorded, accuracy_over_epochs)
plt.xlabel("Centralized Model Accuracy")
plt.ylabel("Epochs")
plt.show()

# loss graph
plt.ylim(0, 1)  # set range
plt.plot(epochs_recorded, loss_over_epochs)
plt.xlabel("Centralized Model Loss")
plt.ylabel("Epochs")
plt.show()
