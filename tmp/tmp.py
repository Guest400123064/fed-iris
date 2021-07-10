#%%
import torch
import torch.nn as nn
import json


class SubModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.sub_linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.sub_linear(x)


class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.sub_module1 = SubModule()
        self.sub_module2 = SubModule()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        
        sub_out1 = self.sub_module1(x)
        sub_out2 = self.sub_module2(x)
        return self.fc(sub_out1 + sub_out2)


#%%
my_model = Module()

d = {k: v.round().cpu().tolist() for k, v in my_model.state_dict().items()}
print(json.dumps(d, indent=True))
# %%


import numpy as np

p = np.load('../round-3-weights.npz', allow_pickle=True)

# %%
