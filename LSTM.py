import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy
from sklearn.metrics import r2_score

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (numpy.mean(numpy.abs((Y_true - Y_pred) / Y_true)) * 100)

def sin_wave(A, f, fs, phi, t):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

def cos_wave(A, f, fs, phi, t):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.cos(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

def get_noise_curve(frequency, time_series):
    noise = np.random.normal(0, time_series, frequency)
    return noise

def data_generation():
    # f=50 hz
    fs = 2000
    y1 = sin_wave(A=30, f=50, fs=fs, phi=0, t=0.5)
    y2 = sin_wave(A=20, f=60, fs=fs, phi=30, t=0.5)
    y3 = cos_wave(A=10, f=80, fs=fs, phi=60, t=0.5)
    y4 = cos_wave(A=5, f=100, fs=fs, phi=90, t=0.5)
    noise  =  get_noise_curve(1000,0.5)
    y = y1 + y2 + y3 + y4 + 3*noise
    x = np.arange(0, 0.5, 1/fs)
    plt.xlabel('t/s')
    plt.ylabel('y')
    plt.grid()
    # plt.plot(x, hz_50, 'k')
    plt.plot(x[0:200], y[0:200], 'k')
    # plt.plot(x, hz_50_30, 'r-.')
    # plt.plot(x, hz_50_60, 'g--')
    # plt.plot(x, hz_50_90, 'b-.')
    # plt.legend(['phase 0', 'phase 30', 'phase 60', 'phase 90'], loc=1)

    return y

def data_to_series_features(data, time_steps):
    data_size = len(data) - time_steps
    series_X = []
    series_y = []
    for i in range(data_size):
        series_X.append(data[i:i + time_steps])
        series_y.append(data[i + time_steps, 0])
    series_X = np.array(series_X)
    series_y = np.array(series_y)
    return series_X, series_y



if __name__ == '__main__':

    dataset = data_generation()
    train, test = dataset[0:800], dataset[800:len(dataset)]
    train = np.reshape(train,[800,1])
    test =  np.reshape(test,[200,1])

    look_back = 10

    trainX, trainY = data_to_series_features(train, look_back)
    testX, testY = data_to_series_features(test, look_back)

    # trainX = numpy.reshape(trainX,[790,10])
    # testX = numpy.reshape(testX,[190,10])

    model = Sequential()  # Sequential为序贯模型，是函数式模型的简略版
    model.add(LSTM(60, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=False))  # 输入维度为1，时间窗的长度为1，隐含层神经元节点个数为120
    model.add(Dropout(0.3))
    # model.add(Dense(24,activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=100, batch_size=5, verbose=2)  # verbose=2 为每个epoch输出一行记录
    model.summary()

    testPredict = model.predict(testX)
    testY = numpy.reshape(testY, [190 , 1])

    rmse = format(RMSE(testY, testPredict), '.4f')
    mape = format(MAPE(testY, testPredict), '.4f')
    r2 = format(r2_score(testY, testPredict), '.4f')
    mae = format(mean_absolute_error(testY, testPredict), '.4f')
    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))

    numpy.save('LSTM.py_predict_artificial.npy',testPredict)