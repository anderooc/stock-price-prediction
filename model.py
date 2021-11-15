from dataset import csvToDataset, historyPoints
import matplotlib

csvPath = input()
ohlcvHistories, nextDayOpen, unscaledY, yNormalizer = csvToDataset(csvPath)

# Percentage of data split for testing
test_split = 0.8
n = int(ohlcvHistories.shape[0] * test_split)

ohlcvTrain = ohlcvHistories[:n]
yTrain = nextDayOpen[:n]

ohlcvTest = ohlcvHistories[n:]
yTest = nextDayOpen[n:]

unscaledYTest = unscaledY[n:]