# https://github.com/nyanp/streamdf/blob/main/streamdf/streamdf.py

from typing import Any, List, Type

import numpy as np
import pandas as pd


class StreamPy:
    """
    Numpy array that aim to improve performance to extend.
    Holds the number of rows currently stored and expands it before it fills up.
    """

    def __init__(
        self,
        values: np.ndarray,  # 2-dim with any size and any dtype
        columns: List[str],
        dtype: Type = None,
        default_value: Any = None,
        length: int = 0,
    ):
        """
        Args:
            values (np.ndarray): Numpy array to store data.
            columns (List[str]): Column name of store data.
            dtype (Type): Data type of store data.
            default_value (int): Value to be inserted when an exception is raised.
            length (int): Number of data currently stored.
        """
        self.values = values
        self.columns = columns
        self.dtype = dtype if dtype is not None else values.dtype
        self.default_value = default_value
        self.length = length

        self.capacity = values.shape[0]

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return self.length

    @classmethod
    def empty(
        cls,
        columns: List[str],
        dtype: Type,
        default_value: Any = None,
        capacity: int = 1000,
    ):
        values = np.zeros((capacity, len(columns)), dtype=dtype)

        return cls(values, columns, default_value=default_value)

    @property
    def shape(self):
        return self.values.shape

    @property
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.columns, dtype=self.dtype)

    def extend(self, array: np.ndarray):
        n_row, n_col = array.shape

        assert n_col == len(
            self.columns
        ), "The number of columns in the array does not match the number of column names."

        if self.length + n_row >= self.capacity:
            self._grow(self.length + n_row)

        # Nan は default_value
        array[pd.isnull(array)] = self.default_value

        self.values[self.length : self.length + n_row] = array
        self.length += n_row

    def _grow(self, min_capacity):
        capacity = max(int(1.5 * self.capacity), min_capacity)
        new_data_len = capacity - self.capacity
        assert new_data_len > 0, "Invalid data length to grow."

        self.values = np.concatenate(
            [
                self.values,
                np.empty((new_data_len, len(self.columns)), dtype=self.values.dtype),
            ]
        )
        self.capacity += new_data_len

    def last_n(self, n: int) -> np.ndarray:
        res = self.values[max(self.length - n, 0) : self.length]

        if len(res) == n:
            return res

        res = np.concatenate([np.zeros((n - len(res), len(self.columns)), dtype=res.dtype), res])
        return res
