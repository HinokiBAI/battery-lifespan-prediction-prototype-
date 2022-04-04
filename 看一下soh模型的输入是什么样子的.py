import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from torch.utils.data.dataloader import DataLoader


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2640, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


model = Net()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0009, amsgrad=False)


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 2641):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    for i in range(len(dataY)):
        if dataY[i].astype("float64") == 0:
            dataY[i] = str(dataY[i - 1][0].astype("float64"))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset, dataY


file_name1 = './data/soh/vltm5.csv'
file_name2 = './data/soh/vltm6.csv'
file_name3 = './data/soh/vltm7.csv'
file_name4 = './data/soh/vltm18.csv'
file_name5 = './data/soh/vltm45.csv'
file_name6 = './data/soh/vltm46.csv'
file_name7 = './data/soh/vltm47.csv'
file_name8 = './data/soh/vltm48.csv'
file_name9 = './data/soh/vltm53.csv'
file_name10 = './data/soh/vltm54.csv'
file_name11 = './data/soh/vltm55.csv'
file_name12 = './data/soh/vltm56.csv'

series5 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True)
series6 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True)
series7 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True)
series18 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True)
series45 = read_csv(file_name5, header=None, parse_dates=[0], squeeze=True)
series46 = read_csv(file_name6, header=None, parse_dates=[0], squeeze=True)
series47 = read_csv(file_name7, header=None, parse_dates=[0], squeeze=True)
series48 = read_csv(file_name8, header=None, parse_dates=[0], squeeze=True)
series53 = read_csv(file_name9, header=None, parse_dates=[0], squeeze=True)
series54 = read_csv(file_name10, header=None, parse_dates=[0], squeeze=True)
series55 = read_csv(file_name11, header=None, parse_dates=[0], squeeze=True)
series56 = read_csv(file_name12, header=None, parse_dates=[0], squeeze=True)

index = []
raw_values5 = series5.values
raw_values6 = series6.values
raw_values7 = series7.values
raw_values18 = series18.values
raw_values45 = series45.values
raw_values46 = series46.values
raw_values47 = series47.values
raw_values48 = series48.values
raw_values53 = series53.values
raw_values54 = series54.values
raw_values55 = series55.values
raw_values56 = series56.values
raw_values = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18, raw_values45, raw_values46,
                            raw_values47, raw_values48, raw_values53, raw_values54, raw_values55, raw_values56),
                            axis=0)
look_back = 2640
dataset, dataY = create_dataset(raw_values, look_back)
dataset_5, dataY_5 = create_dataset(raw_values5, look_back)
dataset_6, dataY_6 = create_dataset(raw_values6, look_back)
dataset_7, dataY_7 = create_dataset(raw_values7, look_back)
dataset_18, dataY_18 = create_dataset(raw_values18, look_back)
dataset_45, dataY_45 = create_dataset(raw_values45, look_back)
dataset_46, dataY_46 = create_dataset(raw_values46, look_back)
dataset_47, dataY_47 = create_dataset(raw_values47, look_back)
dataset_48, dataY_48 = create_dataset(raw_values48, look_back)
dataset_53, dataY_53 = create_dataset(raw_values53, look_back)
dataset_54, dataY_54 = create_dataset(raw_values54, look_back)
dataset_55, dataY_55 = create_dataset(raw_values55, look_back)
dataset_56, dataY_56 = create_dataset(raw_values56, look_back)
train_size_5 = int(dataset_5.shape[0] * 0.7)
train_size_6 = int(dataset_6.shape[0] * 0.7)
train_size_7 = int(dataset_7.shape[0] * 0.7)
train_size_18 = int(dataset_18.shape[0] * 0.7)
train_size_45 = int(dataset_45.shape[0] * 0.7)
train_size_46 = int(dataset_46.shape[0] * 0.7)
train_size_47 = int(dataset_47.shape[0] * 0.7)
train_size_48 = int(dataset_48.shape[0] * 0.7)
train_size_53 = int(dataset_53.shape[0] * 0.7)
train_size_54 = int(dataset_54.shape[0] * 0.7)
train_size_55 = int(dataset_55.shape[0] * 0.7)
train_size_56 = int(dataset_56.shape[0] * 0.7)
# split into train and test sets
train_5, test_5 = dataset_5[0:train_size_5], dataset_5[train_size_5:]
train_6, test_6 = dataset_6[0:train_size_6], dataset_6[train_size_6:]
train_7, test_7 = dataset_7[0:train_size_7], dataset_7[train_size_7:]
train_18, test_18 = dataset_18[0:train_size_18], dataset_18[train_size_18:]
train_45, test_45 = dataset_45[0:train_size_45], dataset_45[train_size_45:]
train_46, test_46 = dataset_46[0:train_size_46], dataset_46[train_size_46:]
train_47, test_47 = dataset_47[0:train_size_47], dataset_47[train_size_47:]
train_48, test_48 = dataset_48[0:train_size_48], dataset_48[train_size_48:]
train_53, test_53 = dataset_53[0:train_size_53], dataset_53[train_size_53:]
train_54, test_54 = dataset_54[0:train_size_54], dataset_54[train_size_54:]
train_55, test_55 = dataset_55[0:train_size_55], dataset_55[train_size_55:]
train_56, test_56 = dataset_56[0:train_size_56], dataset_56[train_size_56:]
train = np.concatenate((train_5, train_6, train_7, train_18, train_45, train_46, train_47, train_48, train_53,
                        train_54, train_55, train_56), axis=0)
# print(train)
print(train.shape)
X, y = train[:, 0:-1], train[:, -1]
X = X.reshape(X.shape[0], 660, 4)
print(X.shape)
print(X)
scaler, train_scaled, test5_scaled = scale(train, test_5)
predictions_train = list()
labels = train[:, -1]
print('Forecasting Training Data')
for i in range(len(train_scaled)):
    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
    X = torch.Tensor(X)
    y_hat = model(X)
    yhat = invert_scale(scaler, X, y_hat)
    predictions_train.append(yhat)
    expected = labels[i]
    print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, float(expected)))
# report performance
rmse_train = sqrt(mean_squared_error(np.array(labels).astype("float64") / 2, np.array(predictions_train) / 2))
print('Train RMSE: %.3f' % rmse_train)
index.append(rmse_train)

