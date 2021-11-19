from dataset import csvToDataset, historyPoints
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

csvPath = input()
ohlcv, nextDayClose, unscaledY, yNormalizer = csvToDataset(csvPath)

# Spltting data into training and testing
# ohlcv is x value, nextDayOpen is y value
# shuffle is set to true as default; we want data to be organized by timestamp
trainX, testX, trainY, testY = train_test_split(ohlcv, nextDayClose.ravel(), test_size = 0.8, shuffle = False)

# Splitting up unscaled y values for future comparison
n = int(ohlcv.shape[0] * 0.8)
unscaledYTest = unscaledY[n:]

regr = MLPRegressor(max_iter=1000).fit(trainX, trainY)
regr.predict(testX[:2])
regr.score(testX, testY)