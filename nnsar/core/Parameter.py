from abc import ABC, abstractmethod
from numpy import deg2rad, abs

class Parameter(ABC):
    @abstractmethod
    def __str__(self):
        return f""


class BasicConst(Parameter):
    def __int__(self, v):
        self.__value = v

    def value(self):
        return self.__value

    def __str__(self):
        return f"const: {self.__value}"


class Angle(Parameter):
    def __init__(self, deg):
        self.__deg = deg
        self.__update()

    def __update(self):
        self.__rad = deg2rad(self.__deg)

    def deg(self):
        return self.__deg

    def rad(self):
        return self.__rad

    def __str__(self):
        return f"angle {self.__deg} | {self.__rad}"


class AzRgPair(Parameter):
    def __init__(self, az, rg):
        self.__azimuth = az
        self.__range = rg

    def az(self):
        return self.__azimuth

    def rg(self):
        return self.__range

    def __str__(self):
        return f"az,rg = {self.__azimuth}, {self.__range}"


class LinearFM(Parameter):
    def __init__(self,t, k, f0):
        self.__time = t
        self.__slope = k
        self.__freq0 = f0

    def __str__(self):
        return f"t,k,f0 = {self.__time}, {self.__slope}, {self.__freq0}"

    def t(self):
        return self.__time

    def k(self):
        return self.__slope

    def f0(self):
        return self.__freq0

    def bw(self): # 距离向带宽;发射信号带宽 (Hz)
        return abs(self.k())*self.t()
        # ^ Textbook:P94