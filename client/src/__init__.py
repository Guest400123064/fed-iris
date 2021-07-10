import os
import pathlib

from typing import Tuple, Union
from collections import OrderedDict

import click
import flwr
import torch
import numpy as np

from .core import (
    IrisTrainAgent, 
    IrisInferAgent,
    IrisBatchDataset,
    IrisInferDataset,
    IrisLoader,
    IrisModel,
    IrisLoss,
    IrisOptimizer
)


class InitUtils:

    DIR_INIT = pathlib.Path(__file__).parent
    DIR_ROOT = DIR_INIT.parent

    DIR_DATA_REPO = os.path.join(DIR_ROOT, 'data')
    DIR_MODEL_REPO = os.path.join(DIR_ROOT, 'models')
    DIR_CONFIG_REPO = os.path.join(DIR_ROOT, 'config')

    FILE_DATA_TRAIN = os.path.join(DIR_DATA_REPO, 'train.csv')
    FILE_DATA_VALID = os.path.join(DIR_DATA_REPO, 'valid.csv')
    FILE_MODEL_PARAM = os.path.join(DIR_MODEL_REPO, 'model_params.npz')

    @classmethod
    def make_loader(
        cls, is_train=True
    ) -> Union[IrisLoader, Tuple[IrisLoader, IrisLoader]]:

        if is_train:
            stream_train = IrisBatchDataset(cls.FILE_DATA_TRAIN)
            stream_valid = IrisBatchDataset(cls.FILE_DATA_VALID)
            return IrisLoader(stream_train), IrisLoader(stream_valid)

        stream = IrisInferDataset()
        return IrisLoader(stream)

    @classmethod
    def make_model(
        cls, is_train=True
    ) -> IrisModel:

        model = IrisModel()
        if is_train:
            return model 

        # Read saved model and load_state_dict()
        model_cache = np.load(cls.FILE_MODEL_PARAM)
        model_param_list = [model_cache[p] for p in model_cache.files]
        params_dict = zip(model.state_dict().keys(), model_param_list)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        return model

    @classmethod
    def make_agent(cls, is_train=True):

        if is_train:
            model = cls.make_model()
            loss_fn = IrisLoss()
            optimizer = IrisOptimizer.init(model.parameters())
            loader_train, loader_valid = cls.make_loader()
            agent = IrisTrainAgent(
                model,
                loss_fn,
                optimizer,
                loader_train,
                loader_valid
            )
            return agent

        agent = IrisInferAgent(
            cls.make_model(False),
            cls.make_loader(False)
        )
        return agent


@click.command()
def infer():
    click.echo('[ INFO ] :: entering inference mode')
    agent = InitUtils.make_agent(False)
    agent.predict()


@click.command()
def train():

    click.echo('[ INFO ] :: entering train mode')
    agent = InitUtils.make_agent()
    flwr.client.start_numpy_client('127.0.0.1:8080', agent)


@click.group()
def cli():
    pass

cli.add_command(infer)
cli.add_command(train)
