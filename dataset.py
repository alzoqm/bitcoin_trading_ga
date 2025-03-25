# dataset.py

import torch
import numpy as np

import pandas as pd
 
def make_dataset(
    data_1m, using_column, window_size,
    entry_pos_list, patience_list
):
    data_tensor = []
    for index in range(len(entry_pos_list)):
        if entry_pos_list[index] == 'short' or entry_pos_list[index] == 'long':
            data = data_1m.iloc[index - window_size - 1:index - 1][using_column].values
            patience = patience_list[index]
            # Check if any value in data is NaN
            if np.isnan(data).any():
                combined_data = (None, None)
            else:
                combined_data = (data, patience)
            data_tensor.append(combined_data)
    return data_tensor


def replace_nan_with_zero(tensor):
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    return tensor
