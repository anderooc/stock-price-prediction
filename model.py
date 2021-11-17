from dataset import csvToDataset, historyPoints
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

csvPath = input()
ohlcv, nextDayOpen, unscaledY, yNormalizer = csvToDataset(csvPath)

# Spltting data into training and testing
# OHLCV
trainX, testX, trainY, testY = train_test_split(ohlcv, nextDayOpen, test_size = 0.8, shuffle = False)

# Splitting up unscaled values for future comparison
n = int(ohlcv.shape[0] * 0.8)
unscaledYTest = unscaledY[n:]


