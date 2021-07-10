from easydict import EasyDict
from collections import OrderedDict
from typing import List, Tuple, Dict

import flwr
import torch
import numpy as np

from .loader import (
    IrisLoader, 
    IrisBatchDataset,
    IrisInferDataset
)
from .graph import IrisModel, IrisOptimizer, IrisLoss


class IrisInferAgent:

    def __init__(self, model, loader, config: EasyDict=EasyDict()):
        
        self.config = config
        self.model = model
        self.loader = loader

    def predict(self):

        self.model.eval()
        for infer_x in self.loader:
            infer_i = self.model.predict(infer_x)  # flwr index
            infer_s = self.loader.itos(infer_i)
            print(f'[ INFO ] :: it might be a < {infer_s} >')


class IrisTrainAgent(flwr.client.NumPyClient):

    def __init__(
        self, 
        model,
        loss_fn,
        optimizer,
        loader_train,
        loader_valid=None,
        config=None
    ):
        self.config = config

        # Initialize all sub-components of models
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Data loaders
        self.loader = EasyDict({
            'train': loader_train,
            'valid': loader_valid or loader_train
        })

        # Initialize counters
        self.log_interval = None
        self.max_epoch = None
        self.cur_epoch = 0
        self.cur_iter = 0

    def _train_one_epoch(self) -> None:

        self.cur_iter = 0
        self.model.train()
        for batch_x, batch_y in self.loader.train:
            self.cur_iter += 1

            self.optimizer.zero_grad()
            batch_p = self.model(batch_x)
            batch_loss_val = self.loss_fn(batch_p, batch_y)
            batch_loss_val.backward()
            self.optimizer.step()

    def _validate(self) -> Tuple[float, int]:

        len_dataset = len(self.loader.valid)
        avg_loss_val = 0

        self.model.eval()
        for batch_x, batch_y in self.loader.valid:

            batch_p = self.model(batch_x)
            batch_loss_val = self.loss_fn(batch_p, batch_y)
            avg_loss_val += batch_loss_val.cpu().item() * (len(batch_y) / len_dataset)
            
        return avg_loss_val, len_dataset

    # -----------------------------------------------------------------------------------------------
    """flwr federated learning API"""
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:

        # Retrieve latest parameters and configuration
        #   from remote training coordinator
        self.set_parameters(parameters)
        self.max_epoch = config.get('max_epoch', 1)
        self.log_interval = config.get('log_interval', 4)
        
        # Training loop
        for i in range(self.max_epoch):
            self.cur_epoch = i
            self._train_one_epoch()

        # Upload updated parameters
        parameters = self.get_parameters()
        len_dataset = len(self.loader.train)
        msg = {}
        
        return parameters, len_dataset, msg

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:

        self.set_parameters(parameters)
        avg_loss_val, len_dataset = self._validate()
        msg = {}

        return avg_loss_val, len_dataset, msg

    def get_parameters(self) -> List[np.ndarray]:

        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
