import sys
from typing import Literal, Optional, Dict, List, Union
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

    def init_logger(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        stdout = logging.StreamHandler()
        stdout.setFormatter(CustomFormatter())
        self._logger.addHandler(stdout)
        self._logger.setLevel(self._args.log_level)

    def __init__(self):
        self._relative_black_list = ['path_targets', 'num_short_segments']
        self._methods = []
        self._ignore_lists['method'] = []
        self._ignore_lists['problem'] = []
        self._ignore_lists['metric'] = []
        self.init_parser()
        self.read_data()
        self.process_data()
        if not self._args.info:
            if self._args.problem:
                self.plot(self._args.problem)
            else:
                for metric in self._metrics:
                    self.plot_metric_accros_problems(metric)
                for problem in self._problems:
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
        self._logger.warning(f"Ignoring methods {self._ignore_lists['method']} upon request")
        self._logger.warning(f"Ignoring problems {self._ignore_lists['problem']} upon request")

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
        data: np.ndarray,
        plot_type : Literal['boxplot', 'violinplot'] = 'violinplot'
    ) -> None:
        if plot_type == 'violinplot':
            plot_pointer = axes.violinplot(
                data,
                showmeans=True,
                showmedians=False,
                showextrema=True,
            )
            for i, body in enumerate(plot_pointer['bodies']):
                color = get_color('latte', 'peach')
                body.set_color(color)
                body.set_edgecolor(color)
            for part in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if part not in plot_pointer:
                    continue
                plot_pointer[part].set_edgecolor(get_color('latte', 'lavender'))
                plot_pointer[part].set_linewidth(1)
        elif plot_type == 'boxplot':
            plot_pointer = axes.boxplot(
                data,
                showmeans=True,
            )



    def get_info(self) -> None:
        print(f"There are {self.nb_problems} problems with {self.nb_methods} methods and {self.nb_metrics} metrics.")
        print(f"The cases have {self.n_cases} different cases and {self.n_repetitions} repetitions.\n")

        print(f"Available Methods: {list(self._indexmap['method'].keys())}")
        print(f"Available Problems: {list(self._indexmap['problem'].keys())}")
        print(f"Available Metrics: {list(self._indexmap['metric'].keys())}")



    def read_data(self):
        with open(self._filename, 'r') as f:
            self._raw_data = json.load(f)
        self._problems= list(self._raw_data.keys())
        self._methods = list(self._raw_data[self._problems[0]].keys())
        #self._methods = [m for m in list(self._raw_data[self._problems[0]].keys()) if m not in self._args.ignore_method]
        self._metrics = list(self._raw_data[self._problems[0]][self._methods[0]].keys())
        new_metrics = []
        for metric in self._metrics:
            if metric in self._relative_black_list:
                continue
            else:
                new_metrics.append(metric)
        for metric in self._relative_black_list:
            new_metrics.append(metric)
        self._metrics = new_metrics
        self.n_cases = max([len(self._raw_data[p][self._methods[0]][self._metrics[0]]) for p in self._problems])
        self.n_repetitions = len(self._raw_data[self._problems[0]][self._methods[0]][self._metrics[0]][0])
        self.create_index_map()

    def create_index_map(self):
        self._indexmap = {}
        self._indexmap['fields'] = {'metric' : 2, 'problem': 0, 'method': 1, 'case': 3, 'repetition': 4}

    @property
    def nb_metrics(self) -> int:
        return len(self._metrics)

    @property
    def nb_methods(self) -> int:
        return len(self._methods)

    @property
    def nb_problems(self) -> int:
        return len(self._problems)

    def methods_names(self, index_list: Optional[List[int]] = None) -> List[str]:
        method_names = []
        if index_list is None:
            index_list = list(range(len(self._methods)))
        for i in index_list:
            method = self._methods[i]
            method_names.append(method.replace('Reducer', ''))
        return method_names

    @property
    def problem_names(self) -> List[str]:
        problem_names = []
        for problem in self._problems:
            problem_names.append(problem)
        return problem_names



    def process_data(self) -> None:
        #self._sorted_data = dict((metric, []) for metric in self._metrics)
        self._data_array = np.empty((
            self.nb_problems,
            self.nb_methods,
            self.nb_metrics,
            self.n_cases,
            self.n_repetitions,
        ))
        self._indexmap['problem'] = dict((problem_name, i) for i, problem_name in enumerate(self._raw_data.keys()))
        for problem_i, problem in enumerate(self._raw_data.values()):
            self._indexmap['method'] = dict((methods_name, i) for i, methods_name in enumerate(problem.keys()))
            for method_i, method in enumerate(problem.values()):
                self._indexmap['metric'] = dict((metric_name, i) for i, metric_name in enumerate(method.keys()))
                for metric_i, metric in enumerate(method.values()):
                    values = np.array(metric, dtype=float)
                    pad_values = np.pad(values, ((0, int(self.n_cases - values.shape[0])), (0, 0)), constant_values=np.nan)
                    self._data_array[problem_i, method_i, metric_i, :, :] = pad_values

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

    def indices(self, fieldtype: str) -> List[int]:
        values = self.__getattribute__(f"_{fieldtype}s")
        filtered_values = [v for v in values if v not in self.ignored(fieldtype)]

        return [self.index(fieldtype, value) for value in filtered_values]

    def get_data_array(self, indices: List[Union[List[int], None, int]], nan: Optional[str] = None) -> np.ndarray:
        idx = []
        for i, index in enumerate(indices):
            if index is None:
                idx.append(list(range(self._data_array.shape[i])))
            elif isinstance(index, int):
                idx.append([index])
            elif isinstance(index, list):
                idx.append(index)
        res = self._data_array[np.ix_(
            idx[0],
            idx[1],
            idx[2],
            idx[3],
            idx[4],
        )]
        if nan == 'zero':
            return np.nan_to_num(res, copy=True, nan=0.0)
        elif nan == 'filter':
            new_shape = list(res.shape)
            new_shape[3] = -1
            return np.reshape(res[np.logical_not(np.isnan(res))], tuple(new_shape))

        return res

    def plot_metric_accros_problems(self, metric: str):
        self._logger.info(f"Creating summary plot for metric {metric}")
        fig, axis = plt.subplots()
        fig.set_size_inches(6, 6)
        reference_method = self._args.compare_to
        method_indices = self.indices('method')
        problem_indices = self.indices('problem')
        raw_data = self.get_data_array([
            problem_indices,
            method_indices,
            self.index('metric', metric),
            None,
            None,
        ], nan='zero')
        data_metric = np.sum(
            np.reshape(
                raw_data,
                (len(problem_indices), len(method_indices), -1),
            ),
            axis = 2,
        ).T
        if reference_method is not None:
            raw_data_reference = self.get_data_array([
                problem_indices,
                self.index('method', reference_method),
                self.index('metric', metric),
                None,
                None,
            ], nan='zero')
            data_metric_reference = np.sum(
                np.reshape(
                    raw_data_reference,
                    (len(problem_indices), 1, -1),
                ),
                axis=2,
            ).T
            relative_change = data_metric / data_metric_reference
            self.create_barplot(axis, relative_change, metric)
        else:
            self.create_barplot(axis, data_metric, metric)
        plt.subplots_adjust(wspace=0.05)
        fig.tight_layout()
        if self._args.output:
            plt.savefig(f"{self._args.foldername}/{metric}_summary.png", dpi=500)
        if self._args.show:
            plt.show()

    def create_barplot(self, axes: Axes, data: np.ndarray, metric: str):
        problem_indices = self.indices('problem')
        method_indices = self.indices('method')
        methods_names = self.methods_names(method_indices)
        x_positions = np.arange(0, len(problem_indices))
        width = 0.9/self.nb_methods
        for method_i, method_index in enumerate(method_indices):
            method_i_positions = x_positions + method_i * width
            axes.bar(method_i_positions, data[method_i, :], width, label=f"{methods_names[method_i]}")
        axes.set_ylabel(f'{metric}')
        axes.set_title(f'{metric} by problem and method')
        axes.set_xticks(x_positions + (len(method_indices) - 1) / 2 * width)
        problem_names = [self._problems[i] for i in problem_indices]
        axes.set_xticklabels(problem_names, rotation=45, fontsize=16, ha="right")
        axes.legend(loc = 'lower left')
        max_value = np.max(data)
        max_value = 1.30
        min_value = np.min(data)
        min_value = 0.1
        offset = 1/100 * (max_value - min_value)
        axes.set_ylim([min_value-offset, max_value+offset])



        
        

    def plot(self, problem: Optional[str] = None):
        self._logger.info(f"Plotting for problem {problem}")
        fig, axs = plt.subplots(1, self.nb_metrics)
        fig.set_size_inches((3 * self.nb_metrics, 6))
        reference_method = self._args.compare_to
        max_value = 1.0
        min_value = 1e5
        if problem is None:
            problem = self._problems[0]
        method_indices = self.indices('method')
        for metric_i, metric in enumerate(self._metrics):
            raw_data = self.get_data_array([
                self.index('problem', problem),
                method_indices,
                self.index('metric', metric),
                None,
                None,
            ], nan='filter')
            data_metric = np.reshape(
                raw_data,
                (len(method_indices), -1),
            ).T
            if reference_method is not None and metric not in self._relative_black_list:
                raw_data_reference = self.get_data_array([
                    self.index('problem', problem),
                    self.index('method', reference_method),
                    self.index('metric', metric),
                    None,
                    None,
                ], nan='filter')
                data_metric_reference = np.reshape(
                    raw_data_reference,
                    (1, -1),
                ).T
                if np.any(data_metric_reference == 0):
                    self._logger.error(
                        f"Zeros in reference method for metric {metric} in problem {problem}: Skipping plotting"
                    )
                    return
                relative_change = data_metric / data_metric_reference
                self.create_statistics_plot(axs[metric_i], relative_change)
                axs[metric_i].plot([0.5, len(method_indices)+0.5], [1.05, 1.05], 'r.-.')
                axs[metric_i].plot([0.5, len(method_indices)+0.5], [0.95, 0.95], 'g.-.')
                axs[metric_i].set_xlim([0.5, 0.5 + len(method_indices)])
                axs[metric_i].set_xticks([])
                max_value = max(max_value, np.max(relative_change))
                min_value = min(min_value, np.min(relative_change))
            else:
                self.create_statistics_plot(axs[metric_i], data_metric)


            method_names = self.methods_names(method_indices)
            axs[metric_i].set_xticks(list(range(1, len(method_names)+1)), method_names, fontsize=18, rotation=90)
            axs[metric_i].set_title(f"{metric}")
        fig.suptitle(f"{problem}")
        if reference_method:
            for metric_i, metric in enumerate(self._metrics):
                if metric in self._relative_black_list:
                    continue
                axs[metric_i].set_ylim([min_value - 0.01, max_value + 0.01])
                if metric_i > 0:
                    axs[metric_i].set_yticklabels([])
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

