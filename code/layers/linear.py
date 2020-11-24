import torch

import utils.saver


class Linear(torch.nn.Module, utils.saver.Saver):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
