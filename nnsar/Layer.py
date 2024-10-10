import numpy as np
import matplotlib.pyplot as plt
import abc
from abc import ABC
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn
#

from nnsar.Sensor import Sensor

class Layer(torch.nn.Module):
    def __init__(self,sensor: Sensor,**kwargs):
        super().__init__(**kwargs)
        self.sensor=sensor

    def forward(self, mat:torch.Tensor):
        return mat