import sys
from typing import Literal, Optional, Dict, List, Tuple, Union
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.axes import Axes
import numpy as np
import logging

import argparse

from science_utils.colors.catppuccin import get_color


matplotlib.use('tkagg')
font = {'size'   : 18}



matplotlib.rc('font', **font)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

def valid_log_level(level):
    try:
        return int(getattr(logging, level))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid log level: {level}")

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    blue = "\x1b[34;20m" 
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_full = "%(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format_light = "%(name)s - %(levelname)s - %(message)s (%(filename)s)"

    FORMATS = {
        logging.DEBUG: grey + format_full + reset,
        logging.INFO: blue + format_light + reset,
        logging.WARNING: yellow + format_full + reset,
        logging.ERROR: red + format_full + reset,
        logging.CRITICAL: bold_red + format_full + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class BenchmarkData():
    _problem_names: List[str]
    _method_names: Dict[str, str]
    _metric_names: Dict[str, Dict[str, str]]
    def __init__(self, filename: str):
        self._filename = filename
        with open(self._filename, 'r') as f:
            self._raw_data = json.load(f)
        self._problem_names = list(self._raw_data.keys())
        self._method_names = {}
        self._metric_names = {}

        for problem_name in self._problem_names:
            self._method_names[problem_name] = list(self._raw_data[problem_name].keys())
        for problem_name in self._problem_names:
            self._metric_names[problem_name] = {}
            for method_name in self._method_names[problem_name]:
                self._metric_names[problem_name][method_name] = list(self._raw_data[problem_name][method_name].keys())


        self._nb_problems = len(self._problem_names)
        self._nb_methods = [len(self._method_names[problem_name]) for problem_name in self._problem_names]
        self._nb_metrics = [
            [
                len(mp[method_name]) for method_name in self._method_names[problem_name]
            ] for problem_name, mp in self._metric_names.items()
        ]
        self._nb_queries = {
                problem_name:
                    len(list(list(self._raw_data[problem_name].values())[0].values())[0][0]) for problem_name in self._problem_names}
        self._nb_repetitions = {
                problem_name:
                    len(list(list(self._raw_data[problem_name].values())[0].values())[0]) for problem_name in self._problem_names}

    def get_data(self, problem_name: str | None = None, method_name: str | None = None, metric_name: str | None = None) -> list:
        if problem_name is None:
            return self._raw_data
        if method_name is None:
            return self._raw_data[problem_name]
        if metric_name is None:
            return self._raw_data[problem_name][method_name]
        return self._raw_data[problem_name][method_name][metric_name]

    def get_data_array(self, 
            problem_names: Union[str, List[str]] = 'all',
            method_names: Union[str, List[str]] = 'all',
            metric_names: Union[str, List[str]] = 'all',
            average: Union[str, None] = None,
        ) -> Tuple[List[np.ndarray], List]:
        """get data as numpy array"""
        if problem_names == 'all':
            problem_names = self._problem_names
        if method_names == 'all':
            method_names = self.common_methods(problem_names)
        if metric_names == 'all':
            metric_names = self.common_metrics(problem_names, method_names)

        result = []
        failures = []
        for problem in problem_names:
            problem_result = []
            problem_failures = []
            for method in method_names:
                method_result = []
                method_failures = []
                for metric in metric_names:
                    data = []
                    failures_temp = 0
                    for i in range(self._nb_repetitions[problem]):
                        assert self._nb_queries[problem] == len(self._raw_data[problem][method][metric][i]), f"Missing queries for problem {problem} method {method} {len(self._raw_data[problem][method][metric][i])} != {self._nb_queries[problem]}"
                        for j in range(self._nb_queries[problem]):
                            if self._raw_data[problem][method][metric][i][j] is None:
                                failures_temp += 1
                                continue
                            else:
                                data.append(float(self._raw_data[problem][method][metric][i][j]))
                    if len(metric_names) == 1:
                        method_result = data
                        method_failures = failures_temp
                    else:
                        method_result.append(data)
                        method_failures.append(failures_temp)
                problem_result.append(method_result)
                problem_failures.append(method_failures)
            result.append(problem_result)
            failures.append(problem_failures)
        if average == 'method':
            averages = []
            for p_i in range(len(result)):
                p_data = []
                for m_j in range(len(result[p_i])):
                    p_data.append(np.mean(np.array(result[p_i][m_j]), axis=0))
                averages.append(np.array(p_data))
            result = averages

        return result, failures

    def common_problems(self) -> List[str]:
        return self._problem_names


    def common_metrics(self,
            problem_names: Union[str, List[str]] = 'all',
            method_names: Union[str, List[str]] = 'all',
        ) -> List[str]:
        """check which metrics are present in every self._metric_names"""
        if problem_names == 'all':
            problem_names = self._problem_names
        if method_names == 'all':
            method_names = self.common_methods(problem_names=problem_names)
        common_metrics = []
        metrics = [self._metric_names[problem_name][method_name] for problem_name in problem_names for method_name in method_names]
        common_metrics = list(set.intersection(*[set(m) for m in metrics]))
        return common_metrics

    def common_methods(self,
            problem_names: Union[str, List[str]] = 'all',
        ) -> List[str]:
        """check which methods are present in every self._method_names"""
        if problem_names == 'all':
            problem_names = self._problem_names
        common_methods = []
        methods = [self._method_names[problem_name] for problem_name in problem_names]
        common_methods = list(set.intersection(*[set(m) for m in methods]))
        return common_methods

    def get_info(self) -> str:
        info = f"Number of problems: {self._nb_problems}\n"
        info += f"Number of methods: {self._nb_methods}\n"
        info += f"Number of metrics: {self._nb_metrics}\n"
        info += f"Number of queries: {self._nb_queries}\n"
        info += f"Number of repetitions: {self._nb_repetitions}\n"
        info += f"Problems: {self._problem_names}\n"
        return info









class BenchPlotter():
    _raw_data: dict
    _data_array: np.ndarray
    _metrics: list
    _methods: list
    _problems: list
    n_cases: int
    n_repetitions: int
    _indexmap: Dict[str, Dict[str, int]]
    _relative_black_list: List[str]
    _parser: argparse.ArgumentParser
    _ignore_lists: Dict[str, List[str]] = {}
    _logger: logging.Logger
    _error_map: Dict[int, str] = {
        1: "Success",
        -2: "Collision",
        99999: "UnknownError"
    }

    def init_logger(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        stdout = logging.StreamHandler()
        stdout.setFormatter(CustomFormatter())
        self._logger.addHandler(stdout)
        self._logger.setLevel(self._args.log_level)

    def __init__(self):
        self._relative_black_list = ['path_targets', 'num_short_segments', 'success']
        self._methods = []
        self._ignore_lists['method'] = []
        self._ignore_lists['problem'] = []
        self._ignore_lists['metric'] = []
        self.init_parser()
        self.read_data()
        if not self._args.info:
            if self._args.problem:
                self.plot(self._args.problem)
            else:
                for metric_name in self._benchmark_data.common_metrics():
                    self.plot_metric_accros_problems(metric_name)
                for problem in self._benchmark_data.common_problems():
                    if problem in self._ignore_lists['problem']:
                        self._logger.warning(f"Ignoring problem {problem} because it is on the black list.")
                        continue
                    self.plot(problem)
        else:
            self.get_info()

    def init_parser(self):
        self._parser = argparse.ArgumentParser(
            prog='PostProcessing of bench data',
            description="Postprocess data from generated as json from the cfree-bench.",
        )
        self._parser.add_argument('--foldername', '-f', help="Folder containing the data.", required=False)
        self._parser.add_argument('--compare_to', '-ct', help="Method to compare to.", required=False, default=None)
        self._parser.add_argument('--ignore-method', '-im', help="Methods to ignore.", required=False, default=[], type=str, nargs="+")
        self._parser.add_argument('--ignore-problem', '-ip', help="Problems to ignore.", required=False, default=[], type=str, nargs="+")
        self._parser.add_argument('--ignore-metric', '-ime', help="Metrics to be ignored", required=False, default=[], type=str, nargs="+")
        self._parser.add_argument('--problem', '-p', help="Problem to evaluate.", required=False, default=None)
        self._parser.add_argument('--info', '-i', help="Gets info on data file.", required=False, action='store_true')
        self._parser.add_argument('--output', '-o', help="Toggle to save to png.", required=False, default=False, action='store_true')
        self._parser.add_argument('--show', '-s', help="Should the plot be shown.", required=False, default=False, action='store_true')
        self._parser.add_argument('--with-image', '-wi', help="Should the image be shown in the figure", required=False, default=False, action="store_true")
        self._parser.add_argument('--log-level', type=valid_log_level, default='INFO',
                    help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
        self._args = self._parser.parse_args()
        self.init_logger()
        self._filename = self._args.foldername + "/data.json"
        self._ignore_lists['method'] = self._args.ignore_method
        self._ignore_lists['problem'] = self._args.ignore_problem
        self._ignore_lists['metric'] = self._args.ignore_metric
        self._logger.warning(f"Ignoring methods {self._ignore_lists['method']} upon request")
        self._logger.warning(f"Ignoring problems {self._ignore_lists['problem']} upon request")
        self._logger.warning(f"Ignoring metrics {self._ignore_lists['metric']} upon request")

    def insert_png(self, axes: Axes, problem_name, position, size):
        image_path = f"{self._args.foldername}/{problem_name}.png"
        arr_img = plt.imread(image_path)
        im = OffsetImage(arr_img, zoom=size)
        ab = AnnotationBbox(
            im,
            position,
            xybox=(0, 0),
            xycoords='data',
            boxcoords="offset points",
            bboxprops=dict(edgecolor='none', facecolor='none')
        )

        axes.add_artist(ab)



    def create_statistics_plot(
        self,
        axes: Axes,
        data: List[np.ndarray],
        plot_type : Literal['boxplot', 'violinplot'] = 'violinplot'
    ) -> None:
        #cleaned_data = [np.array(d)[~np.isnan(d)] for d in data.T]
        cleaned_data = data
        max_data_points = max([d.size for d in cleaned_data])
        if plot_type == 'violinplot':
            plot_pointer = axes.violinplot(
                cleaned_data,
                showmeans=True,
                showmedians=False,
                showextrema=True,
            )
            for i, body in enumerate(plot_pointer['bodies']):
                if cleaned_data[i].size < max_data_points:
                    color = get_color('latte', 'red')
                else:
                    color = get_color('latte', 'green')
                axes.text(
                    i + 1,
                    np.max(cleaned_data[i]),
                    f"{len(cleaned_data[i])}/{max_data_points}",
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=11,
                    color=color,
                )
                body.set_color(color)
                body.set_edgecolor(color)
            for part in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if part not in plot_pointer:
                    continue
                plot_pointer[part].set_edgecolor(get_color('latte', 'lavender'))
                plot_pointer[part].set_linewidth(1)
        elif plot_type == 'boxplot':
            plot_pointer = axes.boxplot(
                cleaned_data,
                showmeans=True,
            )

    def create_success_plot(
        self,
        axes: Axes,
        data: np.ndarray,
    ) -> None:
        cleaned_data = np.array(data, dtype=int)
        unique_flags = np.unique(cleaned_data)
        counts = np.array([np.sum(cleaned_data == flag, axis=0) for flag in unique_flags])
        methods_id = np.arange(self.nb_methods)
        bottom = np.zeros(self.nb_methods)
        N = unique_flags.shape[0]
        for e in range(N):
            axes.bar(
                methods_id+1,
                counts[e, :],
                bottom=bottom,
                label=f"{self._error_map[unique_flags[e]]}"
            )
            bottom += counts[e, :]
        axes.legend()



    def get_info(self) -> None:
        print(self._benchmark_data.get_info())


    def read_data(self) -> None:
        self._benchmark_data = BenchmarkData(self._filename)


    def index(self, fieldtype, fieldvalue) -> int:
        if fieldvalue is None:
            return self._indexmap['fields'][fieldtype]
        try:
            valueindex = self._indexmap[fieldtype][fieldvalue]
        except:
            raise IndexError(f"Index {fieldvalue} does not exist in {fieldtype}. Options {self._indexmap[fieldtype]}")
        return valueindex

    def ignored(self, fieldtype: str) -> List[str]:
        return self._ignore_lists[fieldtype]



    def plot_metric_accros_problems(self, metric: str):
        method_names = self._benchmark_data.common_methods()
        method_names = [m for m in method_names if m not in self._ignore_lists['method']]
        if metric in self._ignore_lists['metric']:
            self._logger.warning(f"Ignoring metric {metric} because it is on the black list.")
            return
        self._logger.info(f"Creating summary plot for metric {metric}")
        fig, axis = plt.subplots()
        fig.set_size_inches(8, 6)
        reference_method = self._args.compare_to
        data, failures = self._benchmark_data.get_data_array(
                problem_names = 'all',
                method_names = method_names,
                metric_names = [metric],
                average = 'method',
        )
        if reference_method is not None:
            raw_data_reference, _ = self._benchmark_data.get_data_array(
                problem_names = 'all',
                method_names = [reference_method],
                metric_names = [metric],
                average = 'method',
            )
            relative_change = np.array(data) / np.array(raw_data_reference)
            self.create_barplot(axis, relative_change, metric)
        else:
            self.create_barplot(axis, np.array(data), metric, failures=failures)
        plt.subplots_adjust(wspace=0.05)
        fig.tight_layout()
        if self._args.output:
            plt.savefig(f"{self._args.foldername}/{metric}_summary.png", dpi=500)
        if self._args.show:
            plt.show()

    def create_barplot(self, axes: Axes, data: np.ndarray, metric: str, failures: Optional[List[int]] = None):
        problem_names = self._benchmark_data.common_problems()
        method_names = self._benchmark_data.common_methods()
        method_names = [m for m in method_names if m not in self._ignore_lists['method']]
        x_positions = np.arange(0, len(problem_names))
        method_indices = list(range(0, len(method_names)))
        width = 0.9/len(method_names)
        for method_i, method_index in enumerate(method_indices):
            method_i_positions = x_positions + method_i * width
            axes.bar(method_i_positions, data[:, method_i], width, label=f"{method_names[method_i]}")
            if failures:
                for j, pos in enumerate(method_i_positions):
                    if failures[j][method_i] > 0:
                        axes.text(
                            pos,
                            data[j, method_i],
                            f"{failures[j][method_i]}",
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=10,
                            color=get_color('latte', 'red'),
                        )
        axes.set_ylabel(f'{metric}')
        axes.text(
            0.02,
            0.98,
            f"Failures",
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=10,
            color=get_color('latte', 'red'),
            transform=axes.transAxes,
        )
        axes.set_xticks(x_positions + (len(method_indices) - 1) / 2 * width)
        axes.set_xticklabels(problem_names, rotation=45, fontsize=16, ha="right")
        axes.legend(loc = 'lower left', fontsize=14, bbox_to_anchor=(0.0, 1.02), ncol=3)
        max_value = np.max(data)
        max_value = 1.50
        min_value = np.min(data)
        min_value = 0.1
        offset = 1/100 * (max_value - min_value)
        #axes.set_ylim([min_value-offset, max_value+offset])



        
        

    def plot(self, problem: str):
        self._logger.info(f"Plotting for problem {problem}")
        metric_names = self._benchmark_data.common_metrics(problem_names=[problem])
        method_names = self._benchmark_data.common_methods(problem_names=[problem])
        metric_names = [m for m in metric_names if m not in self._ignore_lists['metric']]
        method_names = [m for m in method_names if m not in self._ignore_lists['method']]
        fig, axs = plt.subplots(1, len(metric_names))
        if len(metric_names) == 1:
            axs = [axs]
        fig.set_size_inches((3 * len(metric_names), 6))
        reference_method = self._args.compare_to
        max_value = 1.0
        min_value = 1e5
        for metric_i, metric in enumerate(metric_names):
            raw_data, failures = self._benchmark_data.get_data_array(
                problem_names = [problem],
                method_names = method_names,
                metric_names = [metric],
            )
            if len(method_names) > 1:
                raw_data = raw_data[0]
                failures = failures[0]
            data = [np.array(d) for d in raw_data]
            if reference_method is not None and metric not in self._relative_black_list:
                raw_data_reference, _ = self._benchmark_data.get_data_array(
                    problem_names = [problem],
                    method_names = [reference_method],
                    metric_names = [metric],
                )[0]
                if len(method_names) > 1:
                    raw_data_reference = raw_data_reference[0]
                data_metric_reference = np.array(raw_data_reference)
                if np.any(data_metric_reference == 0):
                    self._logger.error(
                        f"Zeros in reference method for metric {metric} in problem {problem}: Skipping plotting"
                    )
                    return
                for i, d in enumerate(data):
                    if d.shape != data_metric_reference.shape:
                        self._logger.error(
                            f"Data shape mismatch for method {method_names[i]} in problem {problem}"
                        )
                        return
                relative_change =  data / data_metric_reference
                relative_change = [d.tolist() for d in relative_change]
                self.create_statistics_plot(axs[metric_i], relative_change)
            else:
                if metric == 'success':
                    self.create_success_plot(axs[metric_i], data)
                else:
                    self.create_statistics_plot(axs[metric_i], data)


            axs[metric_i].set_xticks(list(range(1, len(method_names)+1)), method_names, fontsize=18, rotation=90)
            axs[metric_i].set_title(f"{metric}")
        fig.suptitle(f"{problem}")
        plt.subplots_adjust(wspace=0.05)
        if self._args.with_image:
            self.insert_png(axs[0], problem, (1.0, max_value-1), 0.14)
        fig.tight_layout()

        if self._args.output:
            plt.savefig(f"{self._args.foldername}/{problem}.png", dpi=500)
        if self._args.show:
            plt.show()

def main():
    BenchPlotter()



if __name__ == "__main__":
    main()

