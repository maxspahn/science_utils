import matplotlib.pyplot as plt
from typing import List
import matplotlib
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq, fft

import argparse

from science_utils.colors.catppuccin import get_color


matplotlib.use('tkagg')
font = {'size'   : 25}



matplotlib.rc('font', **font)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

class TimeSeries():

    def __init__(self, filename: str):
        self._filename = filename
        self.read_data()

    @property
    def resolution(self) -> float:
        return self.t[2] - self.t[1]

    def read_data(self):
        self._data = np.genfromtxt(self._filename, delimiter=',', skip_header=2)
        self._name = self._filename.split('.')[-2]

    @property
    def name(self) -> str:
        return self._name

    @property
    def t(self) -> np.ndarray:
        return self._data[:, 0]

    def values(self, index: int) -> np.ndarray:
        return self._data[:, index]

    def analyse(self, start: float, end: float):
        fig, axis = plt.subplots(1, 1)
        start_i = int(start / self.resolution)
        end_i = int(end / self.resolution)
        N = end_i - start_i

        data = self.values(1)[start_i: end_i]

        freqs = fftfreq(data.size, d = self.resolution)
        fft_data = fft(data)
        peak_coefficients = np.argpartition(np.abs(fft_data)[1:N//2], -1)[-1:]
        peak_freqs = freqs[peak_coefficients+1]
        #axis.plot(freqs[1:N//2], 2/N * np.abs(fft_data[1:N//2]))
        period_durations = 1 / peak_freqs
        self.T = period_durations[0]
        print(f"Dominant frequency's period duration : {self.T}s")

        number_periods = int((end - start) / self.T)
        #shaped_data = np.reshape(data, (number_periods, -1))
        shaped_data = np.resize(data, (number_periods, N//number_periods))
        max_values = np.min(shaped_data, axis=1)
        print(max_values)
        damping_ratios = [max_values[i]/max_values[i+1] for i in range(number_periods-1)]
        deltas = np.log(damping_ratios)
        delta = np.log(max_values[0]/max_values[-1])
        zetas = deltas / np.sqrt((np.pi * 2)**2 + deltas**2)
        zeta = delta / np.sqrt((np.pi * 2)**2 + delta**2)
        self.zeta = zeta

        print(f"Estimated damping coeffecient : {self.zeta}")
        axis.plot(shaped_data.T)
        self.zvd(t0=9,A=10)

    def zvd(self, t0: float = 0, A: float = 1):
        t1 = t0 + 0.5 * self.T
        t2 = t0 + self.T
        K = np.exp(self.zeta * np.pi / (np.sqrt(1 - self.zeta**2)))
        A0 = A * K**2 / (1 + K)**2
        A1 = A * K*2 / (1 + K)**2
        A2 = A * K*1 / (1 + K)**2

        print(f"Impulse 0 at {t0} with amplitude {A0}")
        print(f"Impulse 1 at {t1} with amplitude {A1}")
        print(f"Impulse 2 at {t2} with amplitude {A2}")





class TimeSeriesPlot():
    _series: List[TimeSeries]

    def __init__(self):
        self._series = []
        self.init_parser()
        self.process_data()

    def init_parser(self):
        self._parser = argparse.ArgumentParser(
            prog='PostProcessing of Time Series from csv.',
            description="Postprocess data from time series in csv format.",
        )
        self._parser.add_argument('--filenames', '-f', help="Files containing the data.", required=False, type=str, nargs="+")
        self._parser.add_argument('--info', '-i', help="Gets info on data file.", required=False, action='store_true')
        self._parser.add_argument('--output', '-o', help="Toggle to save to png.", required=False, default=False, action='store_true')
        self._parser.add_argument('--show', '-s', help="Should the plot be shown.", required=False, default=False, action='store_true')
        self._parser.add_argument('--fourier', '--fft', help="Perform Fast Fourier Transform.", required=False, default = False, action="store_true")
        self._parser.add_argument('--ax', '-a', help="Name of investigated axes", required=False, default='z')
        self._args = self._parser.parse_args()


    def process_data(self):
        for f in self._args.filenames:
            self._series.append(TimeSeries(f))
        self.plot()
        if self._args.fourier:
            for time_series in self._series:
                time_series.analyse(10, 17)




    def plot(self):
        fig, axis = plt.subplots(1, 1)
        fig.set_size_inches((10, 5))
        for time_series in self._series:
            axis.plot(time_series.t, time_series.values(1), label=time_series.name)
        axis.set_title("Position Frame 6")
        axis.set_ylabel(f"{self._args.ax}-position [m]")
        axis.set_xlabel("Time [s]")
        axis.set_xlim([8.0, 25])
        axis.legend()
        fig.tight_layout()
        if self._args.show:
            plt.show()
        if self._args.output:
            plt.savefig(f"series.png", dpi=500)



def main():
    TimeSeriesPlot()



if __name__ == "__main__":
    main()



