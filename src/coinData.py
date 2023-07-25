"""
Data class for handeling market data, filtering,
Preperation for traininer and simulations
"""
from klines import *
import pandas as pd
import os

def fetch_write_coin(coin_name, trading_invterval, limit):
    klines = grab_history(coin_name, trading_invterval, limit=limit)
    filename = f"./coin_data/{coin_name}_{trading_invterval}"
    if not os.path.exists("./coin_data/"):
        os.makedirs(filename)
    return write_klines_csv(klines, filename)

def load_klines_from_csv(filename):
    return pd.read_csv(filename)


"""
Used for preparing train/test data with a history length
We want to use to previous 60 days of data to predict the next day
Or we want to space each of these training sets 10 days apart
"""
def section_data(x_data, y_data, history_length, step, y_test_len=1):
    y_test_len -= 1 #offset it to be array index
    X = []
    y = []
    for i in range(0, len(x_data) - history_length - y_test_len, step):
        X.append(x_data[i:i+history_length, :])
        y.append(y_data[i+history_length+y_test_len, :])
    return X, y
