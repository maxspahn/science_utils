from typing import List, Optional, Tuple
import numpy as np
from science_utils.utils.conversions import str2date
from science_utils.utils.conversions import timestamp2date

import pprint

class TimeSeries():
    _filename : str
    _color: str
    _name: str
    _header: List[str]

    def __init__(self, filename: str, color: str, as_date: bool = False):
        self._filename = filename
        self._color = color
        self.read_data(as_date=as_date)

    @property
    def resolution(self) -> float:
        return self.t[2] - self.t[1]

    def read_data(self, as_date: bool = False):
        with open(self._filename, 'r') as f:
            self._header = f.readline().split(',')
        pprint.pprint(self._header)
        if as_date:
            converters = {0: str2date}
        else:
            converters = {}
        self._data = np.genfromtxt(
            self._filename,
            delimiter=',',
            converters = converters,
            skip_header=1,
        )
        self._name = self._filename.split('.')[-2]


    @property
    def name(self) -> str:
        return self._name


    def t_as_date(self) -> np.ndarray:
        tss = np.array([timestamp2date(ts) for ts in self.t()])
        return tss

    def max_value(self, index: int, interval: Optional[Tuple[float, float]] = None) -> float:
        return float(np.max(self.values(index, interval=interval)))

    def min_value(self, index: int, interval: Optional[Tuple[float, float]] = None) -> float:
        return float(np.min(self.values(index, interval=interval)))


    @property
    def color(self) -> str:
        return self._color

    @property
    def all_t(self) -> np.ndarray:
        return self._data[:, 0]

    def t(self, interval: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if interval:
            start_i = self.time_index(interval[0])
            end_i = self.time_index(interval[1])
            return self._data[start_i:end_i, 0]
        else:
            return self._data[:, 0]

    def dvdt(self, index:int, interval: Optional[Tuple[float, float]] = None) -> np.ndarray:
        raw_gradients = np.gradient(self.values(index, interval=interval), self.t(interval=interval))
        nan_mask = np.isnan(raw_gradients)
        raw_gradients[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), raw_gradients[~nan_mask])
        return raw_gradients

    def time_index(self, t: float, eps: float = 1e-3) -> int:
        return np.where(abs(self.all_t - t) < eps)[0][0]

    def get_median(self, index: int, interval: Optional[Tuple[float, float]] = None) -> float:
        median = float(np.median(self.values(index, interval=interval)))
        return median

    def values(self, index: int, interval: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if interval:
            start_i = self.time_index(interval[0])
            end_i = self.time_index(interval[1])
            return self._data[start_i:end_i, index]
        else:
            return self._data[:, index]
