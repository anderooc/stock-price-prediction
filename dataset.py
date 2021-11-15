import pandas as pd
from sklearn import preprocessing
import numpy as np

historyPoints = int(input())

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
    ohlcvHistories = np.array([normalized[i: i + historyPoints].copy() for i in range(len(normalized) - historyPoints)])
    nextDayOpenNormalized = np.array([normalized[:, 0][i + historyPoints].copy() for i in range(len(normalized) - historyPoints)])
    nextDayOpenNormalized = np.expand_dims(nextDayOpenNormalized, -1)

    # Predicting open value with past history points
    nextDayOpen = np.array([data[:, 0][i + historyPoints].copy() for i in range(len(data) - historyPoints)])
    nextDayOpen = np.expand_dims(nextDayOpenNormalized, -1)

    # yNormalizer scales values at end out of normalization
    yNormalizer = preprocessing.MinMaxScaler()

    # Verify there is an equal number of x and y values
    assert ohlcvHistories.shape[0] == nextDayOpenNormalized.shape[0]
    return ohlcvHistories, nextDayOpen, nextDayOpen, yNormalizer