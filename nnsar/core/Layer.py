import torch
#

from nnsar.core.Sensor import Sensor

class Layer(torch.nn.Module):
    def __init__(self, sensor: Sensor, **kwargs):
        super().__init__(**kwargs)
        self.sensor = sensor

        self.const_c = sensor.c.value()

    def forward(self, mat: torch.Tensor):
        return mat
