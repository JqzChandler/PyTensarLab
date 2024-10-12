import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
#
from nnsar.core.Sensor import Sensor
from nnsar.core.Layer import Layer


# RC, Textbook - method (2)
class RangeCompression2(Layer):
    def __init__(self, sensor: Sensor, beta: float=-1, **kwargs):
        super().__init__(sensor, **kwargs)

        # ======= STATEMENT =======
        # Comments to be finished
        self.filterH=nn.Parameter(torch.from_numpy(arrPulseF), requires_grad=False)

    def forward(self,mat:torch.Tensor):
        # mat: [B, Az, Rg]
        
        matF=torch.fft.fft(mat,dim=2) # 0-freq at two ends
        matF=matF*self.filterH # Apply MF
        mat=torch.fft.ifft(matF,dim=2) # back to 2d time domain

        return mat


# RC, Textbook - method (3)
class RangeCompression3(Layer):
    def __init__(self, sensor: Sensor, beta: float = -1, **kwargs):
        super().__init__(sensor, **kwargs)
        # Method 3, directly generate MF based on the formula
        BWr_abs = self.sensor.lfm.bw()
        Kr = self.sensor.lfm.k()

        # 1. Generate MF, freq domain
        axf_rg = self.sensor.axis_freq().rg()
        filterH = np.logical_and(axf_rg>-0.5*BWr_abs, axf_rg<0.5*BWr_abs)*np.exp(1j*np.pi*np.power(axf_rg, 2)/Kr)

        # 2. Apply Kaiser window, time domain
        if beta>=0:
            filterT=np.fft.ifft(filterH)
            filterT=np.fft.fftshift(filterT)*np.kaiser(axf_rg.shape[0], beta)
            filterT=np.fft.ifftshift(filterT)
            filterH=np.fft.fft(filterT)

        # 3. MF along range, 0-freq at center, as nn.Parameter
        self.filterH=nn.Parameter(torch.from_numpy(filterH), requires_grad=False)

    def forward(self,mat:torch.Tensor):
        # mat: [B, Az, Rg]

        matF = torch.fft.fft(mat, dim=2)  # FT along range, 0-freq at two ends
        matF = torch.fft.fftshift(matF, dim=2)

        matF = matF*self.filterH  # apply MF

        matF = torch.fft.ifftshift(matF, dim=2)
        mat = torch.fft.ifft(matF, dim=2)  # back to 2d time domain

        return mat


# RCMC
class RCMC(Layer):
    def __init__(self, sensor:Sensor, sinc_kernel_size:int=8, **kwargs):
        super().__init__(sensor, **kwargs)
        self._sinc_kernel_size = sinc_kernel_size

        Rg = self.sensor.data_size.rg()
        ksize = self._sinc_kernel_size

        # ======= STEP-1 AZIMUTH WEIGHTS=======
        vecWeight = np.exp(-1j*2*np.pi*self.sensor.doppler_central_freq_Fnc()*self.sensor.axis_time().az())
        self.azimuth_weight = nn.Parameter(torch.from_numpy(vecWeight), requires_grad=False)

        # ======= STEP-2 AXIS =======

        # azimuth axis in freq domain
        axis_az_freq = self.sensor.axis_freq().az()
        # range axis in spatial domain (representing position, m)
        axis_rg_range = self.sensor.axis_pos().rg()

        # ======= STEP-3 =======
        arrDelta=np.reshape(
            np.power(axis_az_freq,2),(self.sensor.data_size.az(), 1)
        )*np.reshape(
            axis_rg_range,(1, Rg)
        )
        
        # Comments to be finished

        # ======= STEP-4 =======
        # Comments to be finished

        self.sinc_weight=nn.Parameter(torch.from_numpy(arrSincWeight),requires_grad=False)
        self.pad=(int(ksize/2),int(ksize/2-1))

    def forward(self, mat:torch.Tensor):
        # mat: [B, Az, Rg]
        # print(f"RCMC Layer")

        # ======= STEP-1 =======
        # apply azimuth weight, transfer spatial domain along range axis to range-doppler domain
        mat=torch.permute(mat,(0,2,1))
        mat=mat*self.azimuth_weight
        mat=torch.permute(mat,(0,2,1))
        # ft, along azimuth
        matF=torch.fft.fft(mat,dim=1)
        matF=torch.fft.fftshift(matF,dim=1)

        # ======= STEP-4 =======
        # convolution
        # 用卷积的方式实现sinc插值
        matF=F.pad(matF,self.pad,'circular')
        matF=matF.unfold(2,self._sinc_kernel_size,1)
        mat=torch.sum(self.sinc_weight*matF,dim=3)

        return mat


# AC
class AzimuthCompression(Layer):
    def __init__(self,sensor:Sensor,beta:float=-1,**kwargs):
        super().__init__(sensor,**kwargs)

        # ======= STEP-1 AXIS =======

        # azimuth axis in freq domain
        axis_az_freq=self.sensor.axis_freq().az()  # same as freq axis in RCMC
        # range axis in spatial domain (representing position, m)
        axis_rg_range=self.sensor.axis_pos().rg()

        # ======= STEP-2 AZIMUTH MF=======

        # k on azimuth is changing with R0
        azimuth_LFM_k_Ka=2*(self.sensor.speed()**2)*np.power(np.cos(self.sensor.squint_angle.rad()),2)/(self.sensor.get_lambda()* axis_rg_range)
        Ka_reciprocal=1/azimuth_LFM_k_Ka  # get 1/Ka for further usage

        # reshape: 1d vector -> 2d single column array
        axis_az_freq = np.reshape(axis_az_freq,(axis_az_freq.shape[0],1))
        Ka_reciprocal = np.reshape(Ka_reciprocal,(1,Ka_reciprocal.shape[0]))

        # azimuth MF
        filter_H=np.exp( -1j*np.pi*(np.power((axis_az_freq),2)*Ka_reciprocal))

        # ======= STEP-3 =======
        # apply Kaiser window
        if beta>=0:
            filter_H=filter_H.T  # [Rg,Az], because np.fft.*() work on the -1 th dim as default
            filter_H=np.fft.fftshift(np.fft.ifft(filter_H))
            filter_H=filter_H*np.kaiser(axis_az_freq.shape[0],2.5)
            filter_H=np.fft.fft(np.fft.ifftshift(filter_H))
            filter_H=filter_H.T  # [Az,Rg]

        self.filterH=nn.Parameter(torch.from_numpy(filter_H),requires_grad=False)

    def forward(self, mat: torch.Tensor):
        # mat: [B, Az, Rg]

        # NOTE: the input argument 'mat' is RCMC's result which is in freq domain
        # normally it should be named as 'matF'
        # but here I use 'mat' just because forward() function in parent class 'Layer' is
        # using such a default name for this argument

        # ======= STEP-3 =======
        mat=mat*self.filterH
        mat=torch.fft.ifftshift(mat, dim=1)
        mat=torch.fft.ifft(mat, dim=1)
        # ^ now 'mat' is in time domain after ifft

        return mat


# TOGETHER
class RangeDoppler(Layer):
    def __init__(self, sensor:Sensor, range_compression_kaiser_beta:float=0, rcmc_sinc_kernel_size:int=64, azimuth_compression_kaiser_beta:float=0, **kwargs):
        super().__init__(sensor, **kwargs)
        self.range_compression=RangeCompression3(sensor=self.sensor, beta=range_compression_kaiser_beta)
        self.rcmc=RCMC(sensor=self.sensor, sinc_kernel_size=rcmc_sinc_kernel_size)
        self.azimuth_compression=AzimuthCompression(sensor=self.sensor, beta=azimuth_compression_kaiser_beta)

    def forward(self,mat):
        mat=self.range_compression(mat)
        mat=self.rcmc(mat)
        mat=self.azimuth_compression(mat)
        return mat
