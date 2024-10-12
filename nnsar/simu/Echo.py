import numpy as np
import torch
import torch.nn as nn
#
from nnsar.core.Sensor import Sensor
from nnsar.core.Layer import Layer


class PointEcho(Layer):
    def __init__(self,sensor: Sensor,**kwargs):
        super().__init__(sensor, **kwargs)
        self.Az, self.Rg = self.sensor.data_size.az(), self.sensor.data_size.rg()
        self.AR = self.Az*self.Rg

        self.slant_range = self.sensor.get_slant_range_R0()
        self.speed = self.sensor.speed()
        self.beam_width = self.sensor.get_beam_width().rad()

        self.LFM_t = self.sensor.lfm.t()
        self.LFM_f = self.sensor.lfm.f0()
        self.LFM_k = self.sensor.lfm.k()

        angle_lower = self.sensor.squint_angle.rad()-self.beam_width*0.5
        angle_upper = self.sensor.squint_angle.rad()+self.beam_width*0.5
        self.tanCoef0 = np.tan(angle_upper)
        self.tanCoef1 = np.tan(angle_lower)

        matTimeAz, matTimeRg = self.sensor.matrix_time()
        self.matTimeAz = nn.Parameter(torch.from_numpy(matTimeAz), requires_grad=False)
        self.matTimeRg = nn.Parameter(torch.from_numpy(matTimeRg), requires_grad=False)
        matPosAz, matPosRg = self.sensor.matrix_pos()
        self.matPosAz = nn.Parameter(torch.from_numpy(matPosAz), requires_grad=False)
        self.matPosRg = nn.Parameter(torch.from_numpy(matPosRg), requires_grad=False)

    def forward(self, mat: torch.Tensor):
        # mat: [B, N]
        # mat[:,0] --> pos x (m)
        # mat[:,1] --> pos y (m)
        # mat[:,2] --> echo_amp_A0 (>0.0)
        B, nParams = mat.shape
        assert nParams == 3, f"Wrong param size={nParams}, expected to be 3."

        batchX = mat[:, 0:1]  # Size: [B, 1]
        batchY = mat[:, 1:2]  # Size: [B, 1]
        batchAmp = mat[:, 2:3]  # Size: [B, 1]

        locAz = batchY  # Size: [B, 1]
        locRg = batchX.add(self.slant_range)  # Size: [B, 1]

        azTime0 = locAz-locRg*self.tanCoef0  # azPos0 # Size: [B, 1]
        azTime1 = locAz-locRg*self.tanCoef1  # azPos1 # Size: [B, 1]

        azTime0 = azTime0/self.speed  # Size: [B, 1]
        azTime1 = azTime1/self.speed  # Size: [B, 1]
        azTime_mid = 0.5*(azTime0+azTime1)

        # vec --> mat
        batchAmp = batchAmp.view(B,1,1)
        locAz = locAz.view(B,1,1)
        locRg = locRg.view(B,1,1)
        azTime0 = azTime0.view(B,1,1)
        azTime1 = azTime1.view(B,1,1)
        azTime_mid = azTime_mid.view(B,1,1)

        # ------- winAz -------
        winAz = torch.logical_and(torch.le(azTime0,self.matTimeAz),torch.le(self.matTimeAz,azTime1))

        # ------- winRg -------
        matInstSlantRange = torch.pow(locAz-self.matPosAz,2)+torch.pow(locRg,2)
        matInstSlantRange = torch.sqrt(matInstSlantRange)
        matEchoTime = 2*matInstSlantRange/self.const_c
        winRg = torch.le(torch.abs(matEchoTime-self.matTimeRg), 0.5*self.LFM_t)

        # ------- get matEcho -------
        matEcho = winAz*winRg
        matEcho = matEcho*batchAmp

        matEcho = matEcho*torch.exp((-2*1j*np.pi*self.LFM_f)*matEchoTime)
        matEcho = matEcho*torch.exp((1j*np.pi*self.LFM_k)*torch.pow(matEchoTime-self.matTimeRg,2))

        return matEcho