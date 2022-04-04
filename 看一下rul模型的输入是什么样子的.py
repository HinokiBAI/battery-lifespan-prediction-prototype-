import joblib
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from math import sqrt
from sklearn.metrics import mean_squared_error


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU(True),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


model = Net()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0012, amsgrad=False)


def load_dataset(datasource1: str, datasource2: str, datasource3: str, datasource4: str) -> (numpy.ndarray, MinMaxScaler):
    dataframe1 = pandas.read_csv(datasource1, usecols=[1])
    dataframe1 = dataframe1.fillna(method='pad')
    dataset1 = dataframe1.values
    dataset1 = dataset1.astype('float32')
    dataset1 = dataset1[0:50]

    dataframe2 = pandas.read_csv(datasource2, usecols=[1])
    dataframe2 = dataframe2.fillna(method='pad')
    dataset2 = dataframe2.values
    dataset2 = dataset2.astype('float32')
    dataset2 = dataset2[0:50]

    dataframe3 = pandas.read_csv(datasource3, usecols=[1])
    dataframe3 = dataframe3.fillna(method='pad')
    dataset3 = dataframe3.values
    dataset3 = dataset3.astype('float32')
    dataset3 = dataset3[0:50]

    dataframe4 = pandas.read_csv(datasource4, usecols=[1])
    dataframe4 = dataframe4.fillna(method='pad')
    dataset4 = dataframe4.values
    dataset4 = dataset4.astype('float32')

    dataset = numpy.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


def create_dataset(dataset: numpy.ndarray, look_back: int=1) -> (numpy.ndarray, numpy.ndarray):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


datasource5 = r'./data/rul/5-capacity168.csv'
datasource6 = r'./data/rul/6-capacity168.csv'
datasource7 = r'./data/rul/7-capacity168.csv'
datasource18 = r'./data/rul/18-capacity132.csv'

dataset, scaler = load_dataset(datasource5, datasource6, datasource18, datasource7)
joblib.dump(scaler, r'./result/scaler_rul.pickle')

look_back = 30
dataset_x, dataset_y = create_dataset(dataset, look_back)
dataset_x = numpy.concatenate((dataset_x[0:20], dataset_x[50:70], dataset_x[100:120], dataset_x[150:]), axis=0)
dataset_y = numpy.concatenate((dataset_y[0:20], dataset_y[50:70], dataset_y[100:120], dataset_y[150:]), axis=0)
# dataset_x = numpy.reshape(dataset_x, (dataset_x.shape[0], 10, 3))
# print(dataset_x.shape)
# print(dataset_x)
# print(dataset_x[0].shape)
# print(dataset_y.shape)
# print(dataset_y)
# print(dataset_x[19:20, :])
# print(dataset_x[19].shape)
# print(dataset_x[19])
loss_list = []
dataset_x = torch.Tensor(dataset_x)
dataset_y = torch.Tensor(dataset_y)
for _ in trange(150, desc='fitting model\t', mininterval=1.0):
    for i in range(len(dataset_y)):
        y_pred = model(dataset_x[i])
        loss = criterion(y_pred, dataset_y[i])
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

dataset_predict = []
for i in range(len(dataset_y)):
    y_hat = model(dataset_x[i])
    dataset_predict.append(y_hat)

look_back_buffer = dataset_x[19:20, :]
timesteps = 118

forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
    # make prediction with current lookback buffer
    look_back_buffer = torch.Tensor(look_back_buffer)
    # print(look_back_buffer)
    # print(look_back_buffer.shape)
    cur_predict = model(look_back_buffer)
    # add prediction to result
    cur_predict = cur_predict.detach().numpy()
    forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
    # add new axis to prediction to make it suitable as input
    # cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
    # print(cur_predict)
    # print(cur_predict.shape)
    look_back_buffer = look_back_buffer.reshape(1, 30)
    # remove oldest prediction from buffer
    look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
    # concat buffer with newest prediction
    look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
    # look_back_buffer = look_back_buffer.reshape(1, 10, 3)
    dataset_predict = []
for i in range(len(dataset_y)):
    y_hat = model(dataset_x[i])
    dataset_predict.append(y_hat)

dataset_predict = numpy.array(dataset_predict)
dataset_predict = dataset_predict.reshape(1, -1)
dataset = scaler.inverse_transform(dataset)
dataset_predict = scaler.inverse_transform(dataset_predict)
dataset_y = dataset_y.reshape(1, -1)
dataset_y = scaler.inverse_transform(dataset_y)
forecast_predict = scaler.inverse_transform(forecast_predict)
print(dataset_predict)
print(len(dataset_predict))
print(dataset_predict[:, 0])
dataset_score = sqrt(mean_squared_error(dataset_y, dataset_predict))
print(dataset_score)
