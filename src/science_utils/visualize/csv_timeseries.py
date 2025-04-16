import matplotlib.pyplot as plt
import json
import os
import re
from pprint import pprint
from typing import Any, List, Dict, Tuple, Optional
import matplotlib
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq, fft

import argparse

from science_utils.colors.catppuccin import get_color
from science_utils.utils.timeseries import TimeSeries


matplotlib.use('tkagg')
font = {'size'   : 25}



matplotlib.rc('font', **font)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

axis_to_index: Dict[str, int] = {
        'x': 1,
        'y': 2,
        'z': 3,
}
shaping_method: Dict[int, str] = {
        1: "none",
        2: "ZVD",
        3: "ETMn",
}

class OscilatingTimeSeries(TimeSeries):

    def __init__(self, filename: str, color: str, params: Dict[str, Any]):
        self._filename = filename
        self._color = color
        self._params = params
        self.read_data()

    def read_data(self):
        self._data = np.genfromtxt(self._filename, delimiter=',', skip_header=2)
        self._name = self._filename.split('/')[-1].split('.')[-2]

    def params_str(self, black_list: List[str]) -> str:
        r = ""
        for param, value in self._params.items():
            if param in black_list:
                continue
            if param =='shaping_method':
                r += f"{param}: {shaping_method[value]} "
            elif param == 'n_etm' and not self._params['shaping_method'] == 3:
                continue
            else:
                r += f"{param}: {value} "
        return r[:-1]



    def get_settle_time(self, index: int, interval: Tuple[float, float], value: float, bound: float) -> float:
        start_i = self.time_index(interval[0])
        data = self.values(index, interval=interval)
        above_indices = np.where(np.abs(data - value) > bound)[0]
        if above_indices.size < 1:
            return interval[0]
        t_final = self.all_t[start_i + above_indices[-1]]
        return float(t_final)

    def get_rise_time_derivative(self, index: int, interval: List[float], value: float, bound: float) -> float:
        start_i = self.time_index(interval[0])
        end_i = self.time_index(interval[1])
        data = self.dvdt(index)[start_i:end_i]
        above_indices = np.where(np.abs(data - value) < bound)[0]
        if above_indices.size < 1:
            return interval[0]
        t_final = self.all_t[start_i + above_indices[0]]
        return float(t_final)

    def analyse(self, index: int, start: float, end: float):
        fig, axis = plt.subplots(1, 1)
        start_i = int(start / self.resolution)
        end_i = int(end / self.resolution)
        N = end_i - start_i

        data = self.values(index)[start_i: end_i]

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
    _series: List[OscilatingTimeSeries]
    _levels: Dict[str, Dict[float, str]] = {
        'z':{0.0010: 'green', 0.005: 'red'},
        'y': {0.0010: 'green', 0.005: 'red'},
        'x': {0.01: 'green'}
    }
    _colors: List[str] = [
            'peach',
            'lavender',
            'maroon',
            'green',
            'sky',
            'pink',
            'rosewater',
            'mauve',
            'teal',
            'sapphire',
            'text', 
            'flamingo',
            'blue',
            'yellow',
            'red',
            'peach',
            'lavender',
            'maroon',
            'green',
            'sky',
            'pink',
            'rosewater',
            'mauve',
            'teal',
            'sapphire',
            'text', 
    ]
    _black_list_params = ['m_etm', 'zeta']
    _xlimits = (0.0, 20.0)
    _select_parameters = {
        'q1': 0,
        'q2' : 0,
        'q3': 0,
        'a_max': 250,
        'j_max': 5000,
        'shaping_method': 1,
        'period_duration': 0.6,
        'd_vert': 100,
        'c_vert': 2e5,
    }

    def __init__(self):
        self._series = []
        self.init_parser()
        indices = self.get_indices(self._select_parameters)
        self.extend_blacklist_parameters()
        if self._args.info:
            pprint(self.parameters_indices_map())
            return
        self._fig, self._axis = plt.subplots(1, 1)
        self.process_data(indices)

    def extend_blacklist_parameters(self):
        for param, value in self._select_parameters.items():
            if not isinstance(value, list):
                self._black_list_params.append(param)

    def init_parser(self):
        self._parser = argparse.ArgumentParser(
            prog='PostProcessing of Time Series from csv.',
            description="Postprocess data from time series in csv format.",
        )
        self._parser.add_argument('--folder', '-f', help="Folder containing the data.", required=True, type=str)
        self._parser.add_argument('--info', '-i', help="Gets info on data file.", required=False, action='store_true')
        self._parser.add_argument('--output', '-o', help="Toggle to save to png.", required=False, type=str)
        self._parser.add_argument('--show', '-s', help="Should the plot be shown.", required=False, default=False, action='store_true')
        self._parser.add_argument('--fourier', '--fft', help="Perform Fast Fourier Transform.", required=False, default = False, action="store_true")
        self._parser.add_argument('--ax', '-a', help="Name of investigated axes", required=False, default='z')
        self._parser.add_argument('--settletime', '-st', help="Plot settling time", required=False, default=False, action='store_true')
        self._parser.add_argument('--risetime', '-rt', help="Plot rise time", required=False, default=False, action='store_true')
        self._parser.add_argument('--xlimits', '-xl', help="Limit of x axis", required=False, default=None)
        self._args = self._parser.parse_args()
        self._result_folder = self._args.folder
        self.set_csv_files()
        self.load_parameters()
        if self._args.xlimits:
            self._xlimits = tuple([float(x) for x in self._args.xlimits.split(',')])
        self._index = axis_to_index[self._args.ax]

    def set_csv_files(self) -> None:
        self._filenames = []
        for file in os.listdir(self._result_folder):
            if file.endswith(".csv"):
                self._filenames.append(os.path.join(self._result_folder, file))

    def parameters_indices_map(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        mapping = {}
        for filename in self._filenames:
            index_evaluated_param, index_non_evaluated_param = self.extract_indices(filename)
            parameters = self.parameters_by_run(filename)
            if not index_evaluated_param in mapping:
                mapping[index_evaluated_param] = {}
            mapping[index_evaluated_param][index_non_evaluated_param] = parameters
        return mapping

    def get_indices(self, parameters: Dict[str, Any]) -> List[Tuple[int, int]]:
        result = []
        mapping = self.parameters_indices_map()
        for i, value_i in mapping.items():
            for j, value_j in value_i.items():
                valid = True
                for param, value in value_j.items():
                    if not param in parameters:
                        continue
                    if isinstance(parameters[param], list):
                        if not value in parameters[param]:
                            valid = False
                            break
                    else:
                        if parameters[param] != value:
                            valid = False
                            break
                if valid:
                    result.append((i, j))

        return result




    def load_parameters(self):
        sim_settings_file = self._result_folder + "/sim_settings.json"
        exp_settings_file = self._result_folder + "/exp_settings.json"
        with open(sim_settings_file, 'r') as f:
            self._sim_settings = json.load(f)
        with open(exp_settings_file, 'r') as f:
            self._exp_settings = json.load(f)

    def extract_indices(self, filename: str) -> Tuple[int, int]:
        index_evaluated_param, index_non_evaluated_param = [int(m) for m in re.search(r"(\d+)_(\d+)\.csv$", filename).groups()]
        index_non_evaluated_param -= 1
        return index_evaluated_param, index_non_evaluated_param

    def parameters_by_run(self, filename: str) -> Dict[str, Any]:
        index_evaluated_param, index_non_evaluated_param = self.extract_indices(filename)
        parameters = {}
        for param, values in self._exp_settings['non_evaluated_params'].items():
            parameters[param] = values[index_non_evaluated_param]
        for param, values in self._exp_settings['evaluated_params'].items():
            parameters[param] = values[index_evaluated_param]
        return parameters

    def indices_to_filename(self, indices: Tuple[int, int]) -> str:
        return os.path.join(self._result_folder, f"dymola_results_{indices[0]}_{indices[1]+1}.csv")

    def process_data(self, indices: List[Tuple[int, int]]):
        for i, index_pair in enumerate(indices):
            filename = self.indices_to_filename(index_pair)
            color = self._colors[i]
            params = self.parameters_by_run(filename)
            print(params)
            self._series.append(OscilatingTimeSeries(f"{filename}", get_color('latte', color), params=params))
        self._fig.set_size_inches((18, 9))
        if self._args.fourier:
            for time_series in self._series:
                time_series.analyse(10, 17)
        if self._args.risetime:
            self.plot_dvdt()
            self.plot_rising_times()
        else:
            self.plot(interval=self._xlimits)
        if self._args.settletime:
            self.plot_settling_times()
        self.finalize_plotting()

    def plot_zone(self, time_series, interval: Tuple[float, float], value: float, size: float, color: str='black'):
        start_i = time_series.time_index(interval[0])
        end_i = time_series.time_index(interval[1])
        for v in [value+size, value-size]:
            self._axis.plot(
                    [time_series.all_t[start_i], time_series.all_t[end_i]],
                    [v, v],
                    linestyle='--',
                    linewidth=0.5,
                    color=color,
            )
        self._axis.text(interval[0]+0.1, value+size+0.001, f"+- {size}", fontsize=14, color=color)

    def plot_time(self, t: float, y_positions: Tuple[float, float], color: str ='black', t_ref: float = 0.0, font_color: str =
    'black') -> None:
        self._axis.plot(
                    [t, t],
                    [y_positions[0], y_positions[1]],
                    linestyle='--',
                    linewidth=0.9,
                    color=color,
            )

        self._axis.text(t, y_positions[1], f"{np.round(t-t_ref, decimals=2)}s", fontsize=10, color=font_color)

    def add_fixed_parameters(self):
        parameter_string = ""
        for param, value in self._select_parameters.items():
            if not isinstance(value, list):
                if param == 'shaping_method':
                    parameter_string += f"{param} = {shaping_method[value]}\n"
                else:
                    parameter_string += f"{param}= {value}\n"
        self._axis.text(
            0.85,
            0.99,
            parameter_string,
            transform=self._axis.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
        )


    def show_settling_time(
            self,
            time_series: OscilatingTimeSeries,
            interval: Tuple[float, float],
            bound: float,
            color: str = 'black',
            value: Optional[float] = None
        ) -> None:
        if not value:
            bound_value = time_series.get_median(self._index)
        else:
            bound_value = value
        t_settling = time_series.get_settle_time(self._index, interval, bound_value, bound)
        min_value, max_value = self.min_max_value(self._index, interval=interval)
        diff = (max_value - min_value) / 2
        center = (max_value + min_value) / 2
        ratio = min(4 * bound / diff, 0.9)
        y_values = (center - ratio * diff, center + ratio * diff)
        self.plot_time(t_settling, y_values, color=color, font_color=time_series.color, t_ref=interval[0])
        if not value:
            self.plot_zone(time_series, interval, bound_value, bound, color=color)

    def show_rise_time(
            self,
            time_series: OscilatingTimeSeries,
            interval: Tuple[float, float],
            value: float,
            bound: float,
            color: str = 'black') -> None:
        t_settling = time_series.get_rise_time_derivative(self._index, interval, value, bound)
        self.plot_zone(time_series, interval, value, bound, color=color)
        y_values = (value-5*bound, value+5*bound)
        self.plot_time(t_settling, y_values, color=color, t_ref=interval[0], font_color=time_series.color)

    def min_max_value(self, index: int, interval: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        min_value = 1e9
        max_value = -1e9
        medians = []
        for ts in self._series:
            max_value = max(max_value, ts.max_value(index, interval=interval))
            min_value = min(min_value, ts.min_value(index, interval=interval))
            medians.append(ts.get_median(index, interval=interval))
        median = np.median(np.array(medians))
        if abs(min_value - median) > abs(max_value - median):
            max_value = median + abs(min_value - median)
        else:
            min_value = median - abs(max_value - median)
        return min_value, max_value

    def plot_settling_times(self) -> None:
        levels = self._levels[self._args.ax]
        intervals = [self._xlimits]
        for level, level_color in levels.items():
            median = None
            for time_series in self._series:
                for interval in intervals:
                    if interval[0] < self._xlimits[0]:
                        continue
                    if interval[1] > self._xlimits[1]:
                        continue
                #self.plot_time(interval[0], steady_point, 0.01, color='black', t_ref=interval[0])
                    self.show_settling_time(
                        time_series,
                        interval,
                        level,
                        color=level_color,
                        value=median
                    )
                    if not median:
                        median = time_series.get_median(self._index)

    def plot_rising_times(self) -> None:
        levels = {0.05: 'green'}
        steady_point = 1.73
        for time_series in self._series:
            interval = self._xlimits
            for level, level_color in levels.items():
                self.show_rise_time(time_series, interval, steady_point, level, color=level_color)

    def plot_dvdt(self, interval: Optional[Tuple[float, float]] = None):
        for time_series in self._series:
            self._axis.plot(
                time_series.t(interval=interval),
                time_series.dvdt(self._index, interval=interval),
                label=time_series.params_str(self._black_list_params),
                color=time_series.color
            )
        self._axis.set_title("Velocity Frame 6")


    def plot(self, interval: Optional[Tuple[float, float]] = None):
        for time_series in self._series:
            self._axis.plot(
                time_series.t(interval=interval),
                time_series.values(self._index, interval=interval),
                label=time_series.params_str(self._black_list_params),
                color=time_series.color
            )
        self._axis.set_title("Position Frame 6")

    def finalize_plotting(self):
        self._axis.set_ylabel(f"{self._args.ax}-position [m]")
        self._axis.set_xlabel("Time [s]")
        self._axis.set_xlim(self._xlimits)
        self._axis.legend(loc='lower right', fontsize='15')
        self.add_fixed_parameters()
        self._fig.tight_layout()
        if self._args.show:
            plt.show()
        if self._args.output:
            plt.savefig(self._args.output, dpi=400)



def main():
    TimeSeriesPlot()



if __name__ == "__main__":
    main()



