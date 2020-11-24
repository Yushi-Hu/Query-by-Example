import torch

import utils.saver


class Bilinear(torch.nn.Module, utils.saver.Saver):

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()

        self.bilinear = torch.nn.Bilinear(in1_features, in2_features, out_features, bias=bias)

    def forward(self, x,y):
        return self.bilinear(x, y)
