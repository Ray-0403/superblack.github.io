import math
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100)

def create_dataset(dataset, look_back):  # look_back的作用是将时间序列数据转化为监督学习问题，用前一个时间点的值，预测下一个时间点的值
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 12):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i + look_back + 12), 0])
    return np.array(dataX), np.array(dataY)

milk = pd.read_csv('', usecols=[1], engine='python', skipfooter=0)
dataset=milk.values
dataset=dataset.astype('float32')

scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

lookback_window = 36  # months，滑动窗口

train, test = dataset[:, :], dataset[:,:]
trainX, trainY = create_dataset(train, lookback_window)  # 单步预测
testX, testY = create_dataset(test, lookback_window)
trainX = np.reshape(trainX, (trainX.shape[0], 36, 1))  # （样本个数，1，输入的维度）
testX = np.reshape(testX, (testX.shape[0], 36, 1))
print(trainX.shape,trainY.shape,testX.shape,testY.shape)

i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(12, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])
model.summary()
model.compile('adam', 'mae')
print('Train...')
model.fit(trainX, trainY, epochs=100, verbose=2,batch_size=36)

result = model.predict(testX)
result =scaler.inverse_transform(result )
testY=scaler.inverse_transform(result )

#计算得分
rmse = format(RMSE(testY, result ), '.4f')
mape = format(MAPE(testY,result ), '.4f')
r2 = format(r2_score(testY, result ), '.4f')
mae = format(mean_absolute_error(testY,result ), '.4f')
print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))





