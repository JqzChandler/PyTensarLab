import numpy as np

from nnsar.core.Parameter import BasicConst, Angle, AzRgPair, LinearFM

class Sensor:
    def __init__(self,
                 const_c = 3e8,  # light speed
                 # ^ basic setting

                 distance_center=1e6,  # Slant Range (m)
                 speed=5e3,  # Effective Radar Velocity (m/s)
                 squint_angle=0,  # beam squint angle (deg, along azimuth) (-90,90)

                 antenna=(1.5, 15),  # [az,rg] (m)
                 over_samp=(1024, 2.56e7),  # [az,rg] float

                 lfm_t=1e-6,  # [Textbook: Tr] (s)
                 lfm_k=1.5e14,  # [Textbook: Kr] (Hz/s)
                 lfm_f0=2e9,  # [Textbook: f0] (Hz)
                 # ^ above are all level-0 parameters

                 data_size=(1024, 256),  # [az,rg]
                 ):
        self.antenna = AzRgPair(antenna[0],antenna[1])
        self.oversamp = AzRgPair(over_samp[0], over_samp[1])

        # along az
        self.speed = BasicConst(speed)
        self.d_center = BasicConst(distance_center)
        self.squint_angle = Angle(squint_angle)

        # along rg
        self.lfm = LinearFM(lfm_t,lfm_k,lfm_f0)

        # data size, can customize
        self.data_size = AzRgPair(data_size[0],data_size[1])

        self.c = BasicConst(const_c)

    # ------- DERIVIATED PARAMETERS -------------------------------------------------

    def get_lambda(self):  # wave length (m)
        return self.c.value()/self.lfm.f0()

    def get_beam_width(self):
        bw = 0.886*self.get_lambda()/self.antenna.az()  # rad
        bw = np.rad2deg(bw)  # deg
        return Angle(bw)

    def get_slant_range_R0(self):  # Textbook P142: R0 (m)
        return self.d_center.value()*np.cos(self.squint_angle.rad())

    def get_sampling_rates(self) -> AzRgPair:
        az = self.oversamp.az()*self.doppler_band_width_BWa()
        rg = self.oversamp.rg()*self.lfm.bw()
        return AzRgPair(az, rg)

    # ------- LFM EXTENSION -------------------------------------------------

    def lfm_extension_az_k_Ka(self):
        # Ka=2*(self.get_speed_Vr()**2)/(self.get_wave_length_Lambda()*self.get_slant_range_R0())
        Ka = 2*(self.speed.value()**2)*(np.cos(self.squint_angle.rad())**2)/(self.get_lambda()*self.get_slant_range_R0())
        return Ka

    def lfm_sample_size_Nr(self): # 线性调频信号采样点数
        return int(self.lfm.t()*self.get_sampling_rates().rg())

    # ------- DOPPLER -------------------------------------------------

    def doppler_band_width_BWa(self):  # along azimuth (Hz)
        return 0.886*2.0*self.speed.value()*np.cos(self.squint_angle.rad())/self.antenna.az()

    def doppler_central_freq_Fnc(self):  # Fnc (Hz)
        return 2.0*self.speed.value()*np.sin(self.squint_angle.rad())/self.get_lambda()

    def doppler_ambiguity_Mamb(self):  # motion ambiguity (float) [NOT USED]
        PRF = self.get_sampling_rates().az()
        return np.round(self.doppler_central_freq_Fnc()/PRF)

    # ------- UTILS - BASIC TIME -------------------------------------------------

    def base_time(self) -> AzRgPair:
        pos = AzRgPair(0, self.get_slant_range_R0())

        angle_lim0 = self.squint_angle.rad()-self.get_beam_width().rad()*0.5
        angle_lim1 = self.squint_angle.rad()+self.get_beam_width().rad()*0.5

        # Az
        az_start = pos.az()-pos.rg()*np.tan(angle_lim1)
        az_end = pos.az()-pos.rg()*np.tan(angle_lim0)
        az_mid = 0.5*(az_start+az_end)
        t_mid_az = az_mid/self.speed.value()

        # Rg
        if angle_lim0 >= 0:
            rg_start = pos.rg()*np.reciprocal(np.cos(angle_lim0))
            rg_end = pos.rg()*np.reciprocal(np.cos(angle_lim1))
        else:
            rg_start = pos.rg()
            rg_end = pos.rg()*np.reciprocal(np.cos(angle_lim1))
        rg_mid = 0.5*(rg_start+rg_end)
        t_mid_rg = 2*rg_mid/self.c.value()

        return AzRgPair(t_mid_az, t_mid_rg)

    # ------- UTILS - AXIS -------------------------------------------------

    def axis_time(self) -> AzRgPair:
        t_mid = self.base_time()
        samp_rates=self.get_sampling_rates()

        N = self.data_size.az()
        PRF = samp_rates.az()
        step = np.reciprocal(PRF)
        axis_az = np.arange(-N/2, N/2)*step+t_mid.az()

        N=self.data_size.rg()
        Fr = samp_rates.rg()
        step = np.reciprocal(Fr)
        axis_rg = np.arange(-N/2, N/2)*step+t_mid.rg()+0.5*step

        return AzRgPair(axis_az, axis_rg)

    def axis_pos(self) -> AzRgPair:
        axt = self.axis_time()

        az = axt.az()*self.speed.value()
        rg = axt.rg()*self.c.value()*0.5

        return AzRgPair(az, rg)

    def axis_freq(self, shift=False, center: AzRgPair = None) -> AzRgPair:
        samp_rates = self.get_sampling_rates()

        # az
        N=self.data_size.az()
        Fa=samp_rates.az()
        step = Fa/N
        if N%2 == 1:
            bias = 0.5*step
        else:
            bias = 0

        if center is None:
            az = np.arange(-N/2, N/2)*step+bias+self.doppler_ambiguity_Mamb()*Fa  # Liu's diff
        else:
            az = np.arange(-N/2, N/2)*step+bias+center.az()
            # for CSA, use .doppler_central_freq_Fnc() #241009 <- 240708

        if shift:
            az = np.fft.fftshift(az)

        # rg
        N = self.data_size.rg()
        step = samp_rates.rg()/N
        bias = 0.5*step
        if center is None:
            rg = np.arange(-N/2, N/2)*step+bias
        else:
            rg = np.arange(-N/2, N/2)*step+bias+center.rg()

        if shift:
            rg=np.fft.fftshift(rg)

        return AzRgPair(az, rg)

    # ------- UTILS - MATRIX -------------------------------------------------

    def __axis2matrix(self, axis: AzRgPair) -> AzRgPair:
        # rg before rg, meshgrid use the 1st axis as column, in arr it is the 2nd dimension
        az, rg = np.meshgrid(axis.rg(), axis.az())
        return AzRgPair(az, rg)

    def matrix_time(self) -> AzRgPair:
        return self.__axis2matrix(self.axis_time())

    def matrix_pos(self) -> AzRgPair:
        return self.__axis2matrix(self.axis_pos())







