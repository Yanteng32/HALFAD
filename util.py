import os

import torch
from torch import nn


def model_save(model: nn.Module, path):
    print('Saving model...')
    dir_path_temp = os.path.dirname(path)
    if not os.path.exists(dir_path_temp):
        os.makedirs(dir_path_temp)
    torch.save(model.state_dict(), path)

