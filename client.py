import torch
from torch import nn
from NeuralNetworkClass import NeuralNetwork
from data_provider import DataProvider
from torch.utils.data import DataLoader

torch.manual_seed(42)

# client class
class Client:
    def __init__(self, train_data, test_data, init_model: nn.Module, local_epochs: int, optim_type: str,
                 learning_rate: float, loss_function) -> None:
        self.train_data, self.test_data, self.local_model, self.local_epochs = train_data, test_data, init_model, local_epochs
        self.criterion = loss_function  # loss function
        self.local_optim = self.select_optim(optim_type, learning_rate)  # get optim
        self.accuracy_for_global_epoch = 0

    def select_optim(self, optim_type, learning_rate):
        if optim_type == 'Adam':
            optim = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)
        else:
            assert optim_type == 'Adam', 'Only the Adam optimizer is supported at this time'
        return optim

    def train_local_model(self):
        self.local_model.train()
        for epoch in range(self.local_epochs):  # iterate through epochs
            # print(f"Local Epoch {epoch + 1}")
            correct = 0
            total = 0
            for data, target in self.train_data:  # iterate through batches
                self.local_optim.zero_grad()  # zero the gradients
                # print(data, data.float())
                output = self.local_model(data)  # make predictions / compute output
                loss = self.criterion(output.reshape(len(output)), target.reshape(len(target)))  # calculate loss
                loss.backward()  # backward propagation
                self.local_optim.step()  # update optimizer parameters

                pred = torch.round(output)  # get index of max-log probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            # print(f'Train Accuracy: {(correct / total) * 100}%')
            if epoch == self.local_epochs - 1:
                self.accuracy_for_global_epoch = correct / total * 100

    def evaluate_local_model(self) -> tuple:
        test_loss = 0  # total sum of loss
        correct = 0  # count of correct predictions
        total = 0  # count of total predictions
        self.local_model.eval()  # set to evaluation mode
        with torch.no_grad():  # with no gradients
            for data, target in self.test_data:
                # outputs = self.global_model(data)  # gets predictions
                # _, predicted = torch.max(outputs.data, 1)
                # # calculates the max as a (key, value) pair --> only need the value aka predicted variable
                # # _ is a meaningless variable
                # total += target.size(0)  # size of dimension 0
                # correct += predicted.eq(target.view_as(predicted)).sum().item()  # check correctness of guess
                # print(predicted, " ", target)
                output = self.local_model(data)  # compute output
                test_loss += self.criterion(output.reshape(len(output)), target.reshape(len(target)))  # add batch loss to total
                pred = torch.round(output)  # get index of max-log probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        # print(f'Test Accuracy: {(correct / total) * 100}%')
        test_loss /= len(self.test_data)  # compute total test loss
        return correct / total * 100, test_loss


    def upload_local_model(self):
        self.train_local_model()
        return self.local_model

    def download_global_model(self, global_model: nn.Module):
        self.local_model.load_state_dict(global_model.state_dict())  # load global model


#######################

# # CONSTANTS
# NUM_INPUT = 39
# NUM_OUTPUT = 3
# NUM_EPOCHS = 10
#
# # set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # set neural network parameters
# num_layers = 3
# layer_sizes = [NUM_INPUT, 128, 32, NUM_OUTPUT]
# activation_functions = [nn.LeakyReLU(), nn.LeakyReLU(), nn.Softmax()]
# learning_rate = 1e-5
# loss_function = nn.CrossEntropyLoss()
#
# # create NN
# net = NeuralNetwork(num_layers, layer_sizes, activation_functions)
# optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
#
# # create model
# model = net.to(device)
#
#
# # print(model)
#
#
# # train function
# def train(model, device, train_data_loader, loss_function, optim):
#     model.train()  # set model to training mode
#
#     running_loss = 0
#     total = 0
#     correct = 0
#
#     # iterate through batches
#     for batch_index, data in enumerate(train_data_loader):
#         features, labels = data  # separate data into features and label
#         features, labels = features.to(device), labels.to(device)  # move to device
#         features = features.double()  # BUG FIX -- convert features tensor data type to float
#         optim.zero_grad()  # zero gradients for each batch
#         output = model(features)  # get output / predictions
#         loss = loss_function(output, labels)  # compute output loss
#         loss.backward()  # backward propagation -- compute output gradient
#         optim.step()  # adjust learning weights for optimizer
#         pred = torch.empty(0)
#         for row in output:
#             max = row[0]
#             max_index = 0
#             for i in range(len(row)):
#                 if row[i] > max:
#                     max = row[i]
#                     max_index = i
#             pred = torch.cat((pred, torch.tensor([max_index])), 0)
#         # print(pred, labels)
#         correct += pred.eq(labels.view_as(pred)).sum().item()
#         # print(batch_index, " ", loss)
#
#         running_loss += loss.item()
#         total += labels.size(0)
#         # if batch_index == 5: break
#     print(f"Training Accuracy: {correct} / {total}, i.e. {100 * correct / total}")
#     return running_loss / total
#
#
# # test function
# def test(model, device, test_data_loader, loss_function):
#     model.eval()  # set model to evaluation mode
#
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_index, (features, label) in enumerate(test_data_loader):
#             features, label = features.to(device), label.to(device)  # move to device
#             features = features.double()  # BUG FIX -- convert features tensor data type to float
#             output = model(features)  # compute output
#             test_loss += loss_function(output, label).item()  # add batch loss to total
#             pred = output.argmax(dim=1, keepdim=True)  # get index of max-log probability
#             correct += pred.eq(label.view_as(pred)).sum().item()
#             total += label.size(0)
#             # if batch_index == 5: break
#     test_loss /= total  # compute total test loss
#
#     # output test results
#     print(
#         f'Test Set:\nAverage Loss: {test_loss}\nAccuracy = {correct}/{total} i.e. {100 * correct / total}\n')
#
#
# # iterate through epochs
# # for dataset in range(len(train_data_loaders)):
# #     average_loss = 0
# #     for epoch in range(1, NUM_EPOCHS + 1):
# #         loss = train(model, device, train_data_loaders[dataset], epoch)  # train model and return loss
# #         average_loss += loss
# #         # print(f'\tEpoch {epoch} Loss: {loss}')
# #     average_loss /= NUM_EPOCHS
# #     print(f'Dataset {dataset + 1} Loss: {average_loss}')
# #     test(model, device, test_data_loader)
#
# dp = DataProvider('readmission', 130, 64)
# train_data, test_data = dp.train_data, dp.test_data
# train_data = DataLoader(train_data, batch_size=64, shuffle=True)
# test_data = DataLoader(test_data, batch_size=64, shuffle=False)
#
# print(model)
#
# average_loss = 0
# for epoch in range(1, 400):
#     loss = train(model, device, train_data, loss_function, optim)
#     average_loss += loss
#     print(f'\tEpoch {epoch} Loss: {loss}')
#     test(model, device, test_data, loss_function)
#     # for name, param in model.named_parameters():
#     #     print(f'{name}: {param}')
# average_loss /= NUM_EPOCHS
# print(f'Loss: {average_loss}')
