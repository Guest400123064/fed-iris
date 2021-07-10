import os
import sys
import pathlib

import toml
from easydict import EasyDict


def read_config(path_config) -> EasyDict:
    
    c = toml.load(path_config)
    return EasyDict(c)

def disply_config(config: EasyDict) -> None:
    
    print(config)
