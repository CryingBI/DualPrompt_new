import torch
import torch.nn as nn


class TaskClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(768, 512), 
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(512, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 10)).cuda()
        
        # self.classifier = nn.Sequential(nn.Linear(768, 384), 
        #                                 nn.ReLU(inplace=True), 
        #                                 nn.Linear(384, 384),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(384, 10)).cuda()
    def forward(self, x):
        return self.classifier(x)