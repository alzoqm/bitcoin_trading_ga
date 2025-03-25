# dataset.py

import torch
import numpy as np

import pandas as pd

def make_dataset(
    data_1m, data_1d, using_column, using_column_1d, window_size, window_size_1d,
    entry_pos_list, patience_list, use_1d_data
):
    data_tensor = []
    for index in range(len(entry_pos_list)):
        if entry_pos_list[index] == 'short' or entry_pos_list[index] == 'long':
            data = data_1m.iloc[index - window_size - 1:index - 1][using_column].values
            datetime_1m = data_1m.iloc[index - 1]['Close time']
            if use_1d_data:
                start_date_1d = datetime_1m.date()
                start_date_1d = pd.to_datetime(start_date_1d)
                filtered_data_1d = data_1d[data_1d['Close time'] < start_date_1d]
                filtered_data_1d = filtered_data_1d.iloc[-window_size_1d:][using_column_1d].values
            else:
                filtered_data_1d = None
            patience = patience_list[index]
            combined_data = (data, filtered_data_1d, patience) if use_1d_data else (data, patience)
            data_tensor.append(combined_data)
    return data_tensor

def replace_nan_with_zero(tensor):
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    return tensor
