import math
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense,LSTM
from tcn_NC import TCN
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PyEMD import CEEMDAN
from sklearn.preprocessing import MinMaxScaler


def imf_data(data, look_back):
    X1 = []
    for i in range(look_back, len(data)):
        X1.append(data[i - look_back:i])
    X1.append(data[len(data) - 1:len(data)])
    X_train = np.array(X1)
    return X_train

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

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100)


if __name__ == '__main__':
    dataset =  pd.read_csv('', engine = 'python' , skipfooter = 0)
    dataset = dataset.values
    dataset = dataset[:,1:]
    dataset = dataset.astype('float32')

    dataset_1 = dataset[:, 0]
    dataset_1 = dataset_1.reshape([-1,1])
    scaler_1 =MinMaxScaler(feature_range=(0,1))
    dataset_1 =scaler_1.fit_transform(dataset_1)


    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler_2.fit_transform(dataset)

    dataset_2 = dataset[:,1]
    dataset_3 = dataset[:,2]
    dataset_4 = dataset[:,3]
    dataset_5 = dataset[:,4]
    dataset_6 = dataset[:,5]

    trials = 30
    ceemdan = CEEMDAN(trials=trials)

    imfs_1 = ceemdan.ceemdan(dataset_1.reshape(-1),None,8)

    imfs_2 = ceemdan.ceemdan(dataset_2.reshape(-1),None,8)

    imfs_3 = ceemdan.ceemdan(dataset_3.reshape(-1),None,8)

    imfs_4 = ceemdan.ceemdan(dataset_4.reshape(-1),None,8)

    imfs_5 = ceemdan.ceemdan(dataset_5.reshape(-1),None,8)

    imfs_6 = ceemdan.ceemdan(dataset_6.reshape(-1),None,8)

    look_back = 3
    tool = np.zeros([421,1])
    for imf1 in imfs_1:
        data_1 = imf_data(imf1, 1)
        train, test = data_1[0:2182, :], data_1[2182:len(data_1), :]
        testX, testY = data_to_series_features(test, 3)
        testY = testY.reshape(-1,1)
        tool += testY
    a = []
    j = 1
    for (imf1,imf2,imf3,imf4,imf5,imf6) in zip(imfs_1,imfs_2,imfs_3,imfs_4,imfs_5,imfs_6):
        imf1 = imf_data(imf1, 1)
        imf2 = imf_data(imf2, 1)
        imf3 = imf_data(imf3, 1)
        imf4 = imf_data(imf4, 1)
        imf5 = imf_data(imf5, 1)
        imf6 = imf_data(imf6, 1)
        imf_sum = np.hstack((imf1,imf2,imf3,imf4,imf5,imf6))   # 横向拼接：(9,2607,6)

        print('-' * 45)
        print('This is ' + str(j) + ' time(s)')
        print('*' * 45)

        train, test = imf_sum[0:2183, :], imf_sum[2183:len(dataset_1), :]
        trainX, trainY = data_to_series_features(train, 3)
        testX, testY = data_to_series_features(test, 3)
        inp= Input(shape=(3,6))
        m=TCN()(inp)
        m=Dense(1,activation='linear')(m)
        model=Model(inputs=[inp],outputs=[m])
        model.summary()
        model.compile('adam','mae')
        model.fit(trainX,trainY,epochs=100,verbose=2,batch_size=36)
        prediction = model.predict(testX)
        a.append(prediction)
        j += 1

    a = np.array(a)
    prediction = [0.0 for i in range(len(tool))]
    prediction = np.array(prediction)
    prediction = prediction.reshape(421, 1)
    for i in range(421):
        t = 0.0
        for imf_prediction in a:
            t += imf_prediction[i][:]
        prediction[i, :] = t
    prediction = prediction.reshape(421, 1)
    true = scaler_1.inverse_transform(tool)
    prediction = scaler_1.inverse_transform(prediction)

    rmse = format(RMSE(true, prediction), '.4f')
    mape = format(MAPE(true, prediction), '.4f')
    r2 = format(r2_score(true, prediction), '.4f')
    mae = format(mean_absolute_error(true, prediction), '.4f')
    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))