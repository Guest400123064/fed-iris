import os
import pathlib

import click
from typing import List, Tuple, Optional

import flwr
from flwr.common import parameters_to_weights
import numpy as np


DIR_INIT = pathlib.Path(__file__).parent
DIR_ROOT = DIR_INIT.parent
DIR_MODEL_REPO = os.path.join(DIR_ROOT, 'models')
FILE_MODEL_PARAM = os.path.join(DIR_MODEL_REPO, 'model_params.npz')


class IrisStrategy(flwr.server.strategy.FedAvg):
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[flwr.common.Weights]:
        
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:

            # Convert weights back to list of ndarrays
            save_weights, _ = aggregated_weights
            save_weights = parameters_to_weights(save_weights)

            # Save aggregated_weights
            np.savez(FILE_MODEL_PARAM, *save_weights)
        return aggregated_weights


@click.command()
def cli():

    strategy = IrisStrategy()
    flwr.server.start_server(
        "127.0.0.1:8080", 
        config={"num_rounds": 64},
        strategy=strategy
    )
