import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, num_layers, layer_sizes, activation_functions):
        super(NeuralNetwork, self).__init__()
        self.num_layers = num_layers
        self.linear_stack = nn.Sequential()
        for i in range(num_layers):
            self.linear_stack.add_module(f'fc_{i + 1}', nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i == 0:
                self.linear_stack.add_module(f'drop_{i + 1}', nn.Dropout())
            if i < len(activation_functions):
                self.linear_stack.add_module(f'relu_{i + 1}', activation_functions[i])
        # self.loss_function = loss_function
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.linear_stack.apply(self.init_weights)

    def forward(self, x):
        x = self.linear_stack(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
