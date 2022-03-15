import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data_Duo(data_name):
    data=''
    dataset = pd.read_csv(data).dropna()  #读csv数据，并去掉 NAN 值
    data = dataset.values                 #转成了numpy的数组类型
    data = np.delete(data, 0, axis=1)
    data = data.astype('float32')     #数据为浮点型
    scaler = MinMaxScaler()           #归一化
    y_scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(data[:, 1:])
    # scaled_X = data[:, 1:]
    scaled_y = y_scaler.fit_transform(np.expand_dims((data[:, 0]), axis=1))
    scaled_data = np.concatenate((scaled_y, scaled_X), axis=1)
    return scaled_data, y_scaler

def create_data(dataset, look_back): #单
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 12):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i + look_back + 12), 0])
    return np.array(dataX), np.array(dataY)

def data_to_series_features(data, time_steps): #多
    data_size = len(data) - time_steps
    series_X = []
    series_y = []
    for i in range(data_size):
        series_X.append(data[i:i + time_steps])
        series_y.append(data[i + time_steps, 0])
    series_X = np.array(series_X)
    series_y = np.array(series_y)
    return series_X, series_y

def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False


def apply_weight(series_X, weight):
    weight = np.array(weight)
    weighted_series_X = series_X * np.expand_dims(weight, axis=1)
    return weighted_series_X