import torch
from torch import nn
from NeuralNetworkClass import NeuralNetwork
from data_provider import DataProvider
from torch.utils.data import DataLoader

torch.manual_seed(42)


#######################

# # CONSTANTS
NUM_INPUT = 26
NUM_OUTPUT = 3
NUM_EPOCHS = 10

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# set neural network parameters
num_layers = 2
layer_sizes = [NUM_INPUT, 128, NUM_OUTPUT]
activation_functions = [nn.ReLU()]
learning_rate = 5e-3
loss_function = nn.CrossEntropyLoss()

# create NN
net = NeuralNetwork(num_layers, layer_sizes, activation_functions)
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

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
    for batch_index, data in enumerate(train_data_loader):
        features, labels = data  # separate data into features and label
        # features, labels = features.to(device), labels.to(device)  # move to device
        features = features  # BUG FIX -- convert features tensor data type to float
        optim.zero_grad()  # zero gradients for each batch
        output = model(features)  # get output / predictions
        loss = loss_function(output, labels)  # compute output loss
        loss.backward()  # backward propagation -- compute output gradient
        optim.step()  # adjust learning weights for optimizer
        # pred = torch.empty(0)
        # for row in output:
        #     max = row[0]
        #     max_index = 0
        #     for i in range(len(row)):
        #         if row[i] > max:
        #             max = row[i]
        #             max_index = i
        #     pred = torch.cat((pred, torch.tensor([max_index])), 0)
        # # print(pred, labels)
        # correct += pred.eq(labels.view_as(pred)).sum().item()
        pred = torch.max(output, 1)[1].data.squeeze()
        correct += (pred == labels).sum().item()
        # print(batch_index, " ", loss)

        running_loss += loss.item()
        total += labels.size(0)
        # if batch_index == 5: break
    print(f"Training Accuracy: {correct} / {total}, i.e. {100 * correct / total}")
    return running_loss / total


# test function
def test(model, device, test_data_loader, loss_function):
    model.eval()  # set model to evaluation mode

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (features, label) in enumerate(test_data_loader):
            # features, label = features.to(device), label.to(device)  # move to device
            features = features  # BUG FIX -- convert features tensor data type to float
            output = model(features)  # compute output
            test_loss += loss_function(output, label).item()  # add batch loss to total
            pred = torch.max(output, 1)[1].data.squeeze()
            correct += (pred == label).sum().item()
            total += label.size(0)
            # if batch_index == 5: break
    test_loss /= total  # compute total test loss

    # output test results
    print(
        f'Test Set:\nAverage Loss: {test_loss}\nAccuracy = {correct}/{total} i.e. {100 * correct / total}\n')


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

dp = DataProvider('readmission', 130, 64)
train_data, test_data = dp.train_data, dp.test_data
train_data = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = DataLoader(test_data, batch_size=64, shuffle=False)

print(model)

average_loss = 0
for epoch in range(1, 5000 + 1):
    loss = train(model, device, train_data, loss_function, optim)
    average_loss += loss
    if epoch % 50 == 0:
        print(f'\tEpoch {epoch} Loss: {loss}')
        test(model, device, test_data, loss_function)
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')
average_loss /= NUM_EPOCHS
print(f'Loss: {average_loss}')
