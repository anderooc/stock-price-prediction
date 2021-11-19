from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import argparse


def save_dataset(symbol, time_window):
    # My API key
    input = open("apiKey", "r")
    apiKey = input.readline()
    ts = TimeSeries(key=apiKey, output_format='pandas')

    if time_window == 'daily_adj':
        data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
        pprint(data.head(10))
        data.to_csv(f'./{symbol}_daily_adj.csv')


parser = argparse.ArgumentParser()
parser.add_argument('symbol', type=str, help="stock symbol")
parser.add_argument('time_window', type=str, choices=['daily_adj'], help="stock time period")

namespace = parser.parse_args()
save_dataset(**vars(namespace))
