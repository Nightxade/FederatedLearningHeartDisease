import torch
from torch import nn
from NeuralNetworkClass import NeuralNetwork
from torch.utils.data import DataLoader
import pandas as pd
from CreateDatasetClass import CreateDataset

torch.manual_seed(42)


#######################

# CONSTANTS
NUM_INPUT = 5
NUM_OUTPUT = 1
NUM_EPOCHS = 10

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# set neural network parameters
num_layers = 2
layer_sizes = [NUM_INPUT, 8, NUM_OUTPUT]
activation_functions = [nn.ReLU(), nn.ReLU()]
learning_rate = 1e-4
loss_function = nn.L1Loss()

# create NN
net = NeuralNetwork(num_layers, layer_sizes, activation_functions)
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)

# create model
model = net.to(device)


# print(model)


# train function
def train(model, device, train_data_loader, loss_function, optim):
    model.train()  # set model to training mode

    running_loss = 0
    total = 0
    correct = 0

    # iterate through batches
    for batch_index, (features, labels) in enumerate(train_data_loader):
        # features, labels = features.to(device), labels.to(device)  # move to device
        # features = features  # BUG FIX -- convert features tensor data type to float
        optim.zero_grad()  # zero gradients for each batch
        output = model(features)  # get output / predictions
        loss = loss_function(output.reshape(len(output)), labels)  # compute output loss # BUG FIX -- convert output to double
        loss.backward()  # backward propagation -- compute output gradient
        optim.step()  # adjust learning weights for optimizer
        # pred = output.argmax(dim=1, keepdim=True)  # get index of max-log probability
        # print(pred.reshape(len(pred)), output.reshape(len(output)), labels)
        # correct += pred.eq(labels.view_as(pred)).sum().item()
        # print(batch_index, " ", loss)

        print(output.reshape(len(output)), labels)

        running_loss += loss.item()
        total += labels.size(0)
        # if batch_index == 5: break
    # print(f"Training Accuracy: {correct} / {total}, i.e. {100 * correct / total}")
    return running_loss / len(train_data_loader)


# test function
def test(model, device, test_data_loader, loss_function):
    model.eval()  # set model to evaluation mode

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (features, label) in enumerate(test_data_loader):
            # features, label = features.to(device), label.to(device)  # move to device
            # features = features  # BUG FIX -- convert features tensor data type to float
            output = model(features)  # compute output
            test_loss += loss_function(output, label).item()  # add batch loss to total
            # pred = output.argmax(dim=1, keepdim=True)  # get index of max-log probability
            # correct += pred.eq(label.view_as(pred)).sum().item()
            # total += label.size(0)
            # if batch_index == 5: break
    test_loss /= len(test_data_loader)  # compute total test loss

    # output test results
    print(
        f'Test Set:\nAverage Loss: {test_loss}')


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


def transform(dataset):  # transforms dataset to a tensor
    # return torch.from_numpy(dataset)
    return torch.tensor(dataset)


dataset_pd = pd.read_csv('liver_disorders.csv')
dataset_pd = dataset_pd.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data
train_test_index = int(len(dataset_pd.index) / 5 * 4)  # get index for 80:20 split
train_pd = dataset_pd.copy()[:train_test_index:]
test_pd = dataset_pd.copy()[train_test_index::]

print(train_pd.shape)

train_data = CreateDataset(train_pd, 5, transform)
test_data = CreateDataset(test_pd, 5, transform)

train_data = DataLoader(train_data, batch_size=8, shuffle=True)
test_data = DataLoader(test_data, batch_size=8, shuffle=False)

print(model)

average_loss = 0
for epoch in range(1, 400):
    loss = train(model, device, train_data, loss_function, optim)
    average_loss += loss
    print(f'\tEpoch {epoch} Loss: {loss}')
    test(model, device, test_data, loss_function)
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')
# average_loss /= NUM_EPOCHS
# print(f'Loss: {average_loss}')
