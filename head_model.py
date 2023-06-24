import torch
import torch.nn as nn


class TaskClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(768, 256, bias=True), nn.ReLU(inplace=True), nn.Linear(256, 10, bias=True)).cuda()
    def forward(self, x):
        return self.classifier(x)