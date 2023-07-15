import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Server():
    def __init__(self, init_model: nn.Module, test_data,
                 batch_size: int, loss_function) -> None:  # note that in the real world, the clients, not the server, will have the test data
        self.test_data = DataLoader(test_data, batch_size=batch_size,
                                    shuffle=False)  # no need for shuffling since it's testing data
        self.global_model = init_model
        self.criterion = loss_function

    def aggregate_models(self, models_list: list[nn.Module]) -> nn.Module:
        global_dict = self.global_model.state_dict()  # load current global model (key, value) = (layer name, parameters)
        for name, param in global_dict.items():  # name = which layer, param = parameters (weights, biases)
            # calculate averaged weights and biases for the layer and store that for the layer name
            global_dict[name] = torch.mean(torch.stack([models.state_dict()[name] for models in models_list]), dim=0)
        self.global_model.load_state_dict(
            global_dict)  # update global model with new updated model (key, value) = (layer name, new parameters)
        return self.global_model

    def evaluate_model(self) -> tuple:
        test_loss = 0  # total sum of losses
        correct = 0  # count of correct predictions
        total = 0  # count of total predictions
        self.global_model.eval()  # set to evaluation mode
        with torch.no_grad():  # with no gradients
            for data, target in self.test_data:
                # outputs = self.global_model(data)  # gets predictions
                # _, predicted = torch.max(outputs.data, 1)
                # # calculates the max as a (key, value) pair --> only need the value aka predicted variable
                # # _ is a meaningless variable
                # total += target.size(0)  # size of dimension 0
                # correct += predicted.eq(target.view_as(predicted)).sum().item()  # check correctness of guess
                # print(predicted, " ", target)
                output = self.global_model(data)  # compute output
                test_loss += self.criterion(output.reshape(len(output)), target.reshape(len(target)))  # add batch loss to total
                pred = torch.round(output)  # get index of max-log probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        test_loss /= len(self.test_data)  # compute average test loss
        # print(f'Test Accuracy: {(correct / total) * 100}%')
        return correct / total * 100, test_loss
