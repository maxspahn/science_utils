import argparse

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from science_utils.colors.catppuccin import get_color
from science_utils.utils.timeseries import TimeSeries





class CSVPlotter():

    def __init__(self):
        self.init_parser()

    def init_parser(self):
        self._parser = argparse.ArgumentParser(
            prog='Plotter for csv data',
            description="Plotting csv data in bokeh interactive file",
        )
        self._parser.add_argument('--filenames', '-f', help="Files containing the data.", required=False, type=str, nargs="+")
        self._parser.add_argument("--indices", "-i", help="List of plotted indices", nargs="+", type=int)
        self._parser.add_argument("--asdates", "-ad", help="Are timestamp dates?", required=False, default=False, action="store_true")
        self._args = self._parser.parse_args()

    def load_data(self):
        self._timeseries = TimeSeries(self._args.filenames[0], 'red', as_date=self._args.asdates)

    def plot(self):
        if self._args.asdates:
            self.plot_date()
        else:
            self.plot_normal()


    def plot_date(self, fmt: str = "%Y-%m-%d %H:%M"):
        fig, ax = plt.subplots()
        for i in self._args.indices:
            ax.plot(self._timeseries.t_as_date(), self._timeseries.values(i), 'o-', label=self._timeseries.name, color=self._timeseries.color)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_xlabel("Time as Dates")
        ax.grid()
        plt.show()

    def plot_normal(self, fmt: str = "%Y-%m-%d %H:%M"):
        fig, ax = plt.subplots()
        for i in self._args.indices:
            label = self._timeseries._header[i]
            if 'replicator' in label:
                label = 'driving_wheel_input'
            ax.plot(self._timeseries.t(), self._timeseries.values(i), label=label)
        ax.legend()
        ax.set_xlabel("Time as stamps")
        plt.show()




def main():
    plotter = CSVPlotter()
    plotter.load_data()
    plotter.plot()

