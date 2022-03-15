import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense,LSTM,Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PyEMD import CEEMDAN
from sklearn.preprocessing import MinMaxScaler
import pylab as mpl
from tensorflow.keras.models import Sequential


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

def make_model():
    model = Sequential()
    model.add(LSTM(120, activation='relu', input_shape=(3, 6),return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='linear'))
    return model


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

    trials = 20
    ceemdan = CEEMDAN(trials=trials)

    imfs_1 = ceemdan.ceemdan(dataset_1.reshape(-1),None,8)
    print(imfs_1.shape)
    fig1 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig1.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_1, color='tab:blue')
    plt.ylabel('PM2.5', fontsize=18)
    i = 1
    for imf_1 in imfs_1:
        plt.tight_layout()
        fig1.align_ylabels()
        plt.subplot(len(imfs_1) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_1, color='tab:orange')

        if i == 9:
            fig1.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_1, color='tab:purple')
        i += 1
    plt.savefig('PM2.5_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig1.align_labels()
    plt.show()


    imfs_2 = ceemdan.ceemdan(dataset_2.reshape(-1),None,8)
    imfs_3 = ceemdan.ceemdan(dataset_2.reshape(-1),None,8)


    print(imfs_2.shape)
    fig2 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig2.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_2, color='tab:blue')
    plt.ylabel('PM10', fontsize=18)
    i = 1
    for imf_2 in imfs_2:
        plt.tight_layout()
        fig2.align_ylabels()
        plt.subplot(len(imfs_2) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_2, color='tab:orange')
        # if i == 9:
        #     plt.tick_params(labelsize=13)
        #     plt.plot(imf_1, color='tab:olive')
        if i == 9:
            fig2.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_2, color='tab:purple')
        i += 1
    plt.savefig('PM10_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig2.align_labels()
    plt.show()


    imfs_3 = ceemdan.ceemdan(dataset_3.reshape(-1),None,8)
    print(imfs_3.shape)
    fig3 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig3.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_3, color='tab:blue')
    plt.ylabel('O3', fontsize=18)
    i = 1
    for imf_3 in imfs_3:
        plt.tight_layout()
        fig3.align_ylabels()
        plt.subplot(len(imfs_3) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_3, color='tab:orange')
        # if i == 9:
        #     plt.tick_params(labelsize=13)
        #     plt.plot(imf_1, color='tab:olive')
        if i == 9:
            fig3.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_3, color='tab:purple')
        i += 1
    plt.savefig('O3_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig3.align_labels()
    plt.show()

    imfs_4 = ceemdan.ceemdan(dataset_4.reshape(-1),None,8)
    print(imfs_4.shape)
    fig4 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig4.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_4, color='tab:blue')
    plt.ylabel('NO2', fontsize=18)
    i = 1
    for imf_4 in imfs_4:
        plt.tight_layout()
        fig4.align_ylabels()
        plt.subplot(len(imfs_4) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_4, color='tab:orange')
        # if i == 9:
        #     plt.tick_params(labelsize=13)
        #     plt.plot(imf_1, color='tab:olive')
        if i == 9:
            fig4.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_4, color='tab:purple')
        i += 1
    plt.savefig('NO2_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig4.align_labels()
    plt.show()

    imfs_5 = ceemdan.ceemdan(dataset_5.reshape(-1),None,8)
    print(imfs_5.shape)
    fig5 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig5.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_5, color='tab:blue')
    plt.ylabel('SO2', fontsize=18)
    i = 1
    for imf_5 in imfs_5:
        plt.tight_layout()
        fig5.align_ylabels()
        plt.subplot(len(imfs_5) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_5, color='tab:orange')
        # if i == 9:
        #     plt.tick_params(labelsize=13)
        #     plt.plot(imf_1, color='tab:olive')
        if i == 9:
            fig5.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_5, color='tab:purple')
        i += 1
    plt.savefig('SO2_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig5.align_labels()
    plt.show()

    imfs_6 = ceemdan.ceemdan(dataset_6.reshape(-1),None,8)
    print(imfs_6.shape)    #(9,2607)

    fig6 = plt.figure(figsize=(18, 12))
    plt.subplot(10, 1, 1)
    fig6.align_ylabels()
    plt.subplots_adjust(hspace=0.5)
    plt.tick_params(labelsize=13)
    plt.plot(dataset_6, color='tab:blue')
    plt.ylabel('CO', fontsize=18)
    i = 1
    for imf_6 in imfs_6:
        plt.tight_layout()
        fig6.align_ylabels()
        plt.subplot(len(imfs_6) + 1, 1, i + 1)
        plt.ylabel('IMF' + str(i), fontsize=18)
        if i == 1:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:orange')
        if i == 2:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:green')
        if i == 3:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:red')
        if i == 4:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:purple')
        if i == 5:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:brown')
        if i == 6:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:olive')
        if i == 7:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:red')
        if i == 8:
            plt.tick_params(labelsize=13)
            plt.plot(imf_6, color='tab:orange')
        # if i == 9:
        #     plt.tick_params(labelsize=13)
        #     plt.plot(imf_1, color='tab:olive')
        if i == 9:
            fig6.align_ylabels()
            mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.ylabel('Res', fontsize=18)
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('Time/2h', fontsize=18)
            plt.plot(imf_6, color='tab:purple')
        i += 1
    plt.savefig('CO_result_imf.svg',format='svg',dpi=1800)
    plt.tick_params(labelsize=13)
    fig6.align_labels()
    plt.show()

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

        model = make_model()
        model.summary()
        model.compile(loss='mae', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=36, verbose=2)

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