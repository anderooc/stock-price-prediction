import pandas as pd
from sklearn import preprocessing
import numpy as np

# I decided to use past 50 days, can be changed
historyPoints = 100


def csvToDataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)
    data = data.values

    # Data is normalized
    normalizer = preprocessing.MinMaxScaler()
    normalized = normalizer.fit_transform(data)

    # ohlcv stands for open, high, low, close, volume
    # ohlcvHistories is normalized
    ohlcv = np.array([normalized[i: i + historyPoints].copy() for i in range(len(normalized) - historyPoints)])
    ohlcv = ohlcv[:, :, 0]
    nextDayCloseNormalized = np.array(
        [normalized[:, 3][i + historyPoints].copy() for i in range(len(normalized) - historyPoints)])
    nextDayCloseNormalized = np.expand_dims(nextDayCloseNormalized, -1)

    # Predicting close value with past history points
    nextDayClose = np.array([data[:, 3][i + historyPoints].copy() for i in range(len(data) - historyPoints)])
    nextDayClose = np.expand_dims(nextDayClose, -1)

    # yNormalizer scales values at end out of normalization
    yNormalizer = preprocessing.MinMaxScaler()

    # Verify there is an equal number of x and y values
    assert ohlcv.shape[0] == nextDayCloseNormalized.shape[0]
    return ohlcv, nextDayCloseNormalized, nextDayClose, normalizer
