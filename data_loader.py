# data_loader.py

import pandas as pd

def load_data_1m(pickle_path):
    data_1m = pd.read_pickle(pickle_path)
    data_1m['Open'] = data_1m['Open'].astype(float)
    data_1m['High'] = data_1m['High'].astype(float)
    data_1m['Low'] = data_1m['Low'].astype(float)
    data_1m['Close'] = data_1m['Close'].astype(float)
    data_1m['Volume'] = data_1m['Volume'].astype(float)
    return data_1m
