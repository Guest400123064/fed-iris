import os
from typing import Union

import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader
)


class IrisBatchDataset(Dataset):

    COLNAMES = [
        'sep_len', 'sep_wid', 
            'pet_len', 'pet_width', 'type'
    ]
    FLWR_ITOS = [
        'Iris-setosa', 'Iris-versicolor', 
            'Iris-virginica'
    ]
    FLWR_STOI = {
        'Iris-setosa': 0, 
        'Iris-versicolor': 1, 
        'Iris-virginica': 2
    }

    def __init__(self, path):
        
        self.raw = None
        self.factor = None
        self.target = None
        self._load_from_csv(path)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        
        f = self.factor[idx]
        t = self.target[idx]
        return f, t

    def _load_from_csv(self, path):

        if not os.path.exists(path):
            raise Exception('[ ERRR ] :: 404 dataset not found')
        
        # Read and parses cvs
        self.raw = np.loadtxt(path, dtype=str, delimiter=',')

        # Split XY
        target_col = IrisBatchDataset.COLNAMES.index('type')
        factor_col = [
            IrisBatchDataset.COLNAMES.index(c) 
                for c in ['sep_len', 'sep_wid', 'pet_len', 'pet_width']
        ]
        self.target = np.array([IrisBatchDataset.stoi(c) for c in self.raw[:, target_col]])
        self.factor = self.raw[:, list(factor_col)].astype(np.float16)

        # store as torch tensors    
        self.target = torch.from_numpy(self.target).long()
        self.factor = torch.from_numpy(self.factor).float()

    @classmethod
    def stoi(cls, name):

        i = cls.FLWR_STOI.get(name)
        if (i is None):
            raise Exception(f'unrecognized flower name < {name} >')
        return i

    @classmethod
    def itos(cls, idx):

        if (idx < 0 or idx >= len(cls.FLWR_ITOS)):
            raise Exception(f'flower index out of bound')
        n = cls.FLWR_ITOS[idx]
        return n


class IrisInferDataset(IterableDataset):

    COLNAMES = [
        'sep_len', 'sep_wid', 
            'pet_len', 'pet_width', 'type'
    ]
    FLWR_ITOS = [
        'Iris-setosa', 'Iris-versicolor', 
            'Iris-virginica'
    ]
    FLWR_STOI = {
        'Iris-setosa': 0, 
        'Iris-versicolor': 1, 
        'Iris-virginica': 2
    }

    def __iter__(self):

        while True:
            x = self._load_from_cli()
            if x is None:
                break
            yield torch.from_numpy(x).float()

    def _load_from_cli(self) -> Union[np.ndarray, None]:

        cmd = input('[ INFO ] :: press < ENTER > to make prediction; < Q > to exit: ')
        if cmd == 'Q':
            return None

        sep_len = input('[ READ ] :: |-sep_len >>> ')
        sep_wid = input('[ READ ] :: |-sep_wid >>> ')
        ped_len = input('[ READ ] :: |-pet_len >>> ')
        ped_wid = input('[ READ ] :: |-pet_wid >>> ')

        return np.array([sep_len, sep_wid, ped_len, ped_wid], dtype=float)

    @classmethod
    def stoi(cls, name):

        i = cls.FLWR_STOI.get(name)
        if (i is None):
            raise Exception(f'unrecognized flower name < {name} >')
        return i

    @classmethod
    def itos(cls, idx):

        if (idx < 0 or idx >= len(cls.FLWR_ITOS)):
            raise Exception(f'flower index out of bound')
        n = cls.FLWR_ITOS[idx]
        return n


class IrisLoader:
    
    def __init__(self, dataset, config={}):

        self._config = config
        self._dataset = dataset

        if isinstance(self._dataset, IrisInferDataset):
            self._loader = DataLoader(self._dataset)
        else:
            self._loader = DataLoader(self._dataset, **config)

    def __len__(self):
        
        if isinstance(self._dataset, IrisInferDataset):
            return -1
        else:
            return len(self._dataset)

    def __iter__(self):
        return iter(self._loader)

    def stoi(self, name):
        return self._dataset.stoi(name)

    def itos(self, idx):
        return self._dataset.itos(idx)
