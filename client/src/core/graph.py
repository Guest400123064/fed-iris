import torch
import torch.nn as nn
from easydict import EasyDict


class IrisModel(nn.Module):

    def __init__(self, config: EasyDict=EasyDict()):

        # We know exactly that the iris data set 
        #   contains 4 factors and 3 classes; therefore
        #   the model structure is hard-coded
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.config = config

    def forward(self, x):
        
        # Evaluate the linear part, comput three scores
        #   for each one of the three possible classes
        z = self.linear(x)
        return z

    def predict(self, x):

        # Prediction, return class index
        self.linear.eval()
        with torch.no_grad():
            z = self.linear(x)
            p = torch.softmax(z, 1)
            i = torch.argmax(p, 1)
        return i


class IrisLoss(nn.Module):

    def __init__(self, config: EasyDict=EasyDict()):

        # For details, refer to
        #   <torch.nn.CrossEntropyLoss>
        #   official documentations
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.config = config

    def forward(self, pred, label):

        # Input predictions (pred) should
        #   be the scores (linear part) INSTEAD OF
        #   the probabilities (after softmax applied)
        l = self.loss(pred, label)
        return l


class IrisOptimizer:

    @staticmethod
    def init(params, config: EasyDict=EasyDict()):

        o = torch.optim.SGD(
            params, 
            lr=config.get('lr', 0.01),
            momentum=config.get('momentum', 0),
            nesterov=config.get('nesterov', False)
        )
        return o
