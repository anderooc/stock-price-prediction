from dataset import csvToDataset, historyPoints
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

csvPath = input()
ohlcv, nextDayClose, unscaledY, yNormalizer = csvToDataset(csvPath)

# Spltting data into training and testing
# ohlcv is x value, nextDayOpen is y value
# shuffle is set to true as default; we want data to be organized by timestamp
trainX, testX, trainY, testY = train_test_split(ohlcv, nextDayClose.ravel(), test_size=0.1, shuffle=False)

# Splitting up unscaled y values for future comparison
n = int(ohlcv.shape[0] * 0.9)
unscaledYTest = unscaledY[n:]

regr = MLPRegressor(max_iter=1000).fit(trainX, trainY)
yTestPredict = regr.predict(testX)

normalizer = MinMaxScaler()
normalizer = normalizer.fit(unscaledY)
yTestPredict = normalizer.inverse_transform(np.reshape(yTestPredict, (-1 ,1)))

real_mse = np.mean(np.square(unscaledYTest - yTestPredict))
scaled_mse = real_mse / (np.max(unscaledYTest) - np.min(unscaledYTest)) * 100
print(scaled_mse)

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaledYTest[start:end], label='real')
pred = plt.plot(yTestPredict[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()
