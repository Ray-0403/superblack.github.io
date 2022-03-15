import numpy
import math
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(0,len(dataset) - look_back,12):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        b=dataset[(i+look_back):(i+look_back+12), 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (numpy.mean(numpy.abs((Y_true - Y_pred) / Y_true)) * 100)

if __name__ == '__main__':

    # 读取数据
    dataframe = read_csv('', usecols=[1], engine='python', skipfooter=0)
    dataset = dataframe.values  # 将dataframe转化为numpy数组
    dataset = dataset.astype('float32')  # 将整型变为float

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)  # (64044,1)

    # 创建测试集和训练集

    train, valid ,test = dataset[:,:], dataset[:,:],dataset[:,:]
    look_back = 36
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    testX, testY = create_dataset(test, look_back)

    #创建BP神经网络模型
    model = Sequential()
    model.add(Dense(units=120,  activation='sigmoid', input_shape=(trainX.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(units=12,activation='linear'))
    model.summary()
    model.compile(loss='mse',optimizer='adam')
    history = model.fit(trainX, trainY, epochs=100,validation_data=(validX,validY),shuffle=False,batch_size=36, verbose=2)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation' ], loc='upper right')
    plt.show()

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print('训练预测维度：',trainPredict.shape)
    print('测试预测维度：', testPredict.shape)

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
