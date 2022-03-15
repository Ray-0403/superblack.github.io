import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from keras import regularizers
from keras.utils.vis_utils import plot_model
from datetime import datetime

import pandas as pd
import matplotlib.dates as mdate
import pylab as mpl  # 导入中文字体，避免显示乱码




def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 12):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i + look_back + 12), 0])
    return numpy.array(dataX), numpy.array(dataY)

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (numpy.mean(numpy.abs((Y_true - Y_pred) / Y_true)) * 100)

if __name__ == '__main__':

    # 加载数据
    dataframe = read_csv('', usecols=[1], engine='python', skipfooter=0)
    dataset = dataframe.values  # 将dataframe转化为numpy数组
    # 将整型变为float
    dataset = dataset.astype('float32')

    # 数据处理，归一化至0~1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 划分训练集和测试集
    train, test = dataset[0:13152, :], dataset[13152:len(dataset), :]

    # 创建测试集和训练集
    look_back = 36
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # 调整输入数据的格式  LSTM的输入为[samples,timesteps,features] 样本数，时间步，数据纬度
    trainX = numpy.reshape(trainX, (trainX.shape[0], 36, 1))
    testX = numpy.reshape(testX, (testX.shape[0], 36,1))

    # 创建LSTM神经网络模型
    model = Sequential()  # Sequential为序贯模型，是函数式模型的简略版
    model.add(LSTM(60, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(12,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    history = model.fit(trainX, trainY, epochs=100, batch_size=36, verbose=2)  # verbose=2 为每个epoch输出一行记录

    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    testPredict=numpy.reshape(testPredict,[220*12,1])
    testY=numpy.reshape(testY,[220*12,1])

    # 计算得分
    rmse = format(RMSE(testY, testPredict), '.4f')
    mape = format(MAPE(testY, testPredict), '.4f')
    r2 = format(r2_score(testY, testPredict), '.4f')
    mae = format(mean_absolute_error(testY, testPredict), '.4f')
    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))