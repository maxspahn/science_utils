import matplotlib.pyplot as plt
from typing import List, Dict
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

    def __init__(self, filename: str, color: str):
        self._filename = filename
        self._color = color
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
    def color(self) -> str:
        return self._color

    @property
    def t(self) -> np.ndarray:
        return self._data[:, 0]

    def dvdt(self, index:int) -> np.ndarray:
        return np.gradient(self.values(index), self.t)


    def time_index(self, t: float, eps: float = 1e-3) -> int:
        return np.where(abs(self.t - t) < eps)[0][0]


    def values(self, index: int) -> np.ndarray:
        return self._data[:, index]

    def get_settle_time(self, interval: List[float], value: float, bound: float) -> float:
        start_i = self.time_index(interval[0])
        end_i = self.time_index(interval[1])
        data = self.values(1)[start_i:end_i]
        above_indices = np.where(np.abs(data - value) > bound)[0]
        if above_indices.size < 1:
            return interval[0]
        t_final = self.t[start_i + above_indices[-1]]
        return float(t_final)

    def get_rise_time_derivative(self, interval: List[float], value: float, bound: float) -> float:
        start_i = self.time_index(interval[0])
        end_i = self.time_index(interval[1])
        data = self.dvdt(1)[start_i:end_i]
        above_indices = np.where(np.abs(data - value) < bound)[0]
        if above_indices.size < 1:
            return interval[0]
        t_final = self.t[start_i + above_indices[0]]
        return float(t_final)

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
    _steady_points: Dict[str, float] = {
        'z': 2.03385, 
        'y': 0.00363, 
        'x': 0.0, 
    }
    _levels: Dict[str, Dict[float, str]] = {
        'z':{0.0003: 'green', 0.0005: 'red'},
        'y': {0.0015: 'green', 0.003: 'red'},
        'x': {0.01: 'green'}
    }
    _colors: List[str] = ['peach', 'lavender', 'maroon', 'green', 'sky', 'pink']
    #_colors: List[str] = ['red', 'blue']
    _xlimits = None

    def __init__(self):
        self._series = []
        self.init_parser()
        self._fig, self._axis = plt.subplots(1, 1)
        self.process_data()

    def init_parser(self):
        self._parser = argparse.ArgumentParser(
            prog='PostProcessing of Time Series from csv.',
            description="Postprocess data from time series in csv format.",
        )
        self._parser.add_argument('--filenames', '-f', help="Files containing the data.", required=False, type=str, nargs="+")
        self._parser.add_argument('--info', '-i', help="Gets info on data file.", required=False, action='store_true')
        self._parser.add_argument('--output', '-o', help="Toggle to save to png.", required=False, type=str)
        self._parser.add_argument('--show', '-s', help="Should the plot be shown.", required=False, default=False, action='store_true')
        self._parser.add_argument('--fourier', '--fft', help="Perform Fast Fourier Transform.", required=False, default = False, action="store_true")
        self._parser.add_argument('--ax', '-a', help="Name of investigated axes", required=False, default='z')
        self._parser.add_argument('--settletime', '-st', help="Plot settling time", required=False, default=False, action='store_true')
        self._parser.add_argument('--risetime', '-rt', help="Plot rise time", required=False, default=False, action='store_true')
        self._parser.add_argument('--xlimits', '-xl', help="Limit of x axis", required=False, default=None)
        self._args = self._parser.parse_args()
        if self._args.xlimits:
            self._xlimits = [float(x) for x in self._args.xlimits.split(',')]


    def process_data(self):
        for i, f in enumerate(self._args.filenames):
            color = self._colors[i]
            self._series.append(TimeSeries(f"{f}_{self._args.ax}.csv", get_color('latte', color)))
        self._fig.set_size_inches((18, 10))
        if self._args.fourier:
            for time_series in self._series:
                time_series.analyse(10, 17)
        if self._args.risetime:
            self.plot_dvdt()
            self.plot_rising_times()
        else:
            self.plot()
        if self._args.settletime:
            self.plot_settling_times()
        self.finalize_plotting()

    def plot_zone(self, time_series, interval: List[float], value: float, size: float, color: str='black'):
        start_i = time_series.time_index(interval[0])
        end_i = time_series.time_index(interval[1])
        for v in [value+size, value-size]:
            self._axis.plot(
                    [time_series.t[start_i], time_series.t[end_i]],
                    [v, v],
                    linestyle='--',
                    linewidth=0.5,
                    color=color,
            )
        self._axis.text(interval[0]-0.1, value+size, f"+- {size}", fontsize=8, color=color)

    def plot_time(self, t: float, value: float, bound: float, color: str ='black', t_ref: float = 0.0, font_color: str =
    'black') -> None:
        self._axis.plot(
                    [t, t],
                    [value-bound, value+bound],
                    linestyle='--',
                    linewidth=0.9,
                    color=color,
            )

        self._axis.text(t, value+bound, f"{np.round(t-t_ref, decimals=2)}", fontsize=10, color=font_color)

    def show_settling_time(
            self,
            time_series: TimeSeries,
            interval: List[float],
            value: float,
            bound: float,
            color: str = 'black') -> None:
        t_settling = time_series.get_settle_time(interval, value, bound)
        self.plot_zone(time_series, interval, value, bound, color=color)
        self.plot_time(t_settling, value, bound*4.34, color=color, font_color=time_series.color, t_ref=interval[0])

    def show_rise_time(
            self,
            time_series: TimeSeries,
            interval: List[float],
            value: float,
            bound: float,
            color: str = 'black') -> None:
        t_settling = time_series.get_rise_time_derivative(interval, value, bound)
        self.plot_zone(time_series, interval, value, bound, color=color)
        self.plot_time(t_settling, value, bound*2.34, color=color, t_ref=interval[0], font_color=time_series.color)

    def plot_settling_times(self) -> None:
        steady_point = self._steady_points[self._args.ax]
        levels = self._levels[self._args.ax]
        intervals = [[7.5, 12], self._xlimits]
        for time_series in self._series:
            for interval in intervals:
                if interval[0] < self._xlimits[0]:
                    continue
                if interval[1] > self._xlimits[1]:
                    continue
                #self.plot_time(interval[0], steady_point, 0.01, color='black', t_ref=interval[0])
                for level, level_color in levels.items():
                    self.show_settling_time(time_series, interval, steady_point, level, color=level_color)

    def plot_rising_times(self) -> None:
        params = [[[7.5, 12], 1.03]]
        levels = {0.02: 'green'}
        for time_series in self._series:
            for param in params:
                steady_point = param[1]
                interval = param[0]
                if interval[0] < self._xlimits[0]:
                    continue
                if interval[1] > self._xlimits[1]:
                    continue
                self.plot_time(interval[0], steady_point, 0.05, color='black', t_ref=interval[0])
                for level, level_color in levels.items():
                    self.show_rise_time(time_series, interval, steady_point, level, color=level_color)

    def plot_dvdt(self):
        for time_series in self._series:
            self._axis.plot(time_series.t, time_series.dvdt(1), label=time_series.name, color=time_series.color)
        self._axis.set_title("Velocity Frame 6")


    def plot(self):
        for time_series in self._series:
            self._axis.plot(time_series.t, time_series.values(1), label=time_series.name, color=time_series.color)
        self._axis.set_title("Position Frame 6")

    def finalize_plotting(self):
        self._axis.set_ylabel(f"{self._args.ax}-position [m]")
        self._axis.set_xlabel("Time [s]")
        self._axis.set_xlim(self._xlimits)
        self._axis.legend(loc='lower right')
        self._fig.tight_layout()
        if self._args.show:
            plt.show()
        if self._args.output:
            plt.savefig(self._args.output, dpi=400)



def main():
    TimeSeriesPlot()



if __name__ == "__main__":
    main()



