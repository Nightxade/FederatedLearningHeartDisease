import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_provider import DataProvider
from NeuralNetworkClass import NeuralNetwork
from client import Client
from server import Server

if __name__ == "__main__":
    ##### Get arguments #####
    args = Config().args  # read args
    dataset, num_clients, epochs, global_batch_size = args.dataset, args.nclients, args.epoch, args.global_batch_size  # get global model args
    local_epochs, local_batch_size, optim_type, learning_rate = args.local_epoch, args.local_batch_size, args.optim, args.learning_rate  # get local model args
    loss_function = nn.BCELoss()  # set loss function

    ##### Get dataset #####
    data_provider = DataProvider(dataset, num_clients, local_batch_size)
    clients_train_data = data_provider.split_data_to_clients(data_provider.train_data, num_clients)  # get list of clients' training datasets
    clients_test_data = data_provider.split_data_to_clients(data_provider.test_data, num_clients)  # get list of clients' testing datasets

    ##### Build model #####
    if dataset == 'heart_disease':
        # CONSTANTS
        NUM_INPUT = 13
        NUM_OUTPUT = 1

        # set neural network parameters
        num_layers = 2
        layer_sizes = [NUM_INPUT, 32, NUM_OUTPUT]
        activation_functions = [nn.LeakyReLU(inplace=True), nn.Sigmoid()]

        # create NN
        initial_model = NeuralNetwork(num_layers, layer_sizes, activation_functions)
    else:
        assert dataset == 'heart_disease', 'Only the heart disease dataset is currently supported'

    ##### Initialize clients #####
    clients_list: list[Client] = [Client(clients_train_data[i], clients_test_data[i], initial_model, local_epochs, optim_type, learning_rate, loss_function) for i in range(num_clients)]
    # if you want to differentiate epochs, optimizer types, or learning rates between clients, you can use lists instead
    # list comprehension is preferred because the IDE knows that the list items are Client objects

    ##### Initialize server #####
    server = Server(initial_model, data_provider.test_data, global_batch_size, loss_function)

    ##### Train model
    global_accuracies_over_epochs = []  # y-axis
    global_losses_over_epochs = []  # y-axis
    local_accuracies_over_epochs = []  # y-axis
    local_losses_over_epochs = []  # y-axis
    epochs_recorded = []  # x-axis
    for epoch in range(epochs):
        local_models = [client.upload_local_model() for client in clients_list]  # upload client models
        global_model = server.aggregate_models(local_models)  # aggregate client models

        # for name, param in global_model.named_parameters():
        #     print(f'{name}: {param}')

        evaluation_accuracy, evaluation_loss = server.evaluate_model()  # test / evaluate model

        # in all epochs, record testing accuracies
        if epoch < epochs:
            global_accuracies_over_epochs.append(round(evaluation_accuracy, 2))  # append global model's accuracy
            global_losses_over_epochs.append(evaluation_loss)  # append global model's loss
            client_evaluations = [client.evaluate_local_model() for client in clients_list]  # evaluate clients
            local_accuracies_over_epochs.append([round(client_accuracy, 2) for client_accuracy, client_loss in client_evaluations])  # append accuracy
            local_losses_over_epochs.append([client_loss for client_accuracy, client_loss in client_evaluations])  # append loss

            epochs_recorded.append(epoch + 1)  # append epoch number

        # every 10 epochs, print testing accuracy of global model
        if epoch % 10 == 9:
            print(f'Epoch {epoch + 1}')
            print(f'Test Accuracy: {evaluation_accuracy}%')
        for client in clients_list:
            client.download_global_model(global_model)  # each client gets global model

    ##### Plotting

    plt.style.use('Solarize_Light2')  # set plot style

    ## global accuracies
    plt.ylim(0, 100)  # set range
    plt.plot(epochs_recorded, global_accuracies_over_epochs)
    plt.xlabel("Global Model Accuracy")
    plt.ylabel("Epochs")
    plt.show()

    ## global losses
    plt.ylim(0, 1)  # set range
    plt.plot(epochs_recorded, global_losses_over_epochs)
    plt.xlabel("Global Model Loss")
    plt.ylabel("Epochs")
    plt.show()

    ## local accuracies
    plt.ylim(0, 100)  # set range
    local_accuracies_over_epochs = np.array(local_accuracies_over_epochs).T  # reshape array to switch rows and columns

    ## BAR
    # plot_size = math.ceil(math.sqrt(num_clients))
    # fig, axs = plt.subplots(plot_size, plot_size, sharex=True, sharey=True)  # create square grid of subplots
    # width = 5
    #
    # for client_index, accuracies in enumerate(local_accuracies_over_epochs):  # iterate through all clients
    #     plot_x_index = client_index // plot_size
    #     plot_y_index = client_index % plot_size
    #     # print([epoch + width * multiplier for epoch in epochs_recorded], accuracies)
    #     axs[plot_x_index, plot_y_index].set_title(f"Client {client_index}")
    #     axs[plot_x_index, plot_y_index].bar(epochs_recorded, accuracies, width)
    #
    # # set axis labels
    # for ax in axs.flat:
    #     ax.set(xlabel='Epochs', ylabel='Testing Accuracy')
    #
    # # enable axis label sharing
    # for ax in axs.flat:
    #     ax.label_outer()

    ## LINE
    for client_index, accuracy in enumerate(local_accuracies_over_epochs):
        plt.plot(epochs_recorded, accuracy, label=f"Client {client_index + 1}")
    plt.legend()
    plt.xlabel("Local Models' Accuracies")
    plt.ylabel("Epochs")
    plt.show()

    ## local losses
    plt.ylim(0, 1)  # set range
    local_losses_over_epochs = np.array(local_losses_over_epochs).T  # reshape array to switch rows and columns

    ## BAR
    # plot_size = math.ceil(math.sqrt(num_clients))
    # fig, axs = plt.subplots(plot_size, plot_size, sharex=True, sharey=True)  # create square grid of subplots
    # width = 5
    #
    # for client_index, losses in enumerate(local_losses_over_epochs):  # iterate through all clients
    #     plot_x_index = client_index // plot_size
    #     plot_y_index = client_index % plot_size
    #     # print([epoch + width * multiplier for epoch in epochs_recorded], accuracies)
    #     axs[plot_x_index, plot_y_index].set_title(f"Client {client_index}")
    #     axs[plot_x_index, plot_y_index].bar(epochs_recorded, losses, width)
    #
    # # set axis labels
    # for ax in axs.flat:
    #     ax.set(xlabel='Epochs', ylabel='Testing Loss')
    #
    # # enable axis label sharing
    # for ax in axs.flat:
    #     ax.label_outer()
    #
    # plt.show()

    ## LINE
    for client_index, loss in enumerate(local_losses_over_epochs):
        plt.plot(epochs_recorded, loss, label=f"Client {client_index + 1}")
    plt.legend()
    plt.xlabel("Local Models' Losses")
    plt.ylabel("Epochs")
    plt.show()
