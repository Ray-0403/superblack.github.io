from utils import (load_data_Duo, data_to_series_features,
                   apply_weight, is_minimum)
from algorithm import (initialize_weights, individual_to_key,
                       pop_to_weights, select, reconstruct_population)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model
import argparse
from sklearn.metrics import r2_score
import math
import numpy as np
from model import make_model
from copy import copy
import random
from sklearn.model_selection import train_test_split


def MAPE(Y_true, Y_pred):
    return (np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100)

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting") #实验的具体参数

    parser.add_argument('--iterations', type=int, default=5,               #进化的迭代次数
                        help="Specify the number of evolution iterations")

    parser.add_argument('--batch_size', type=int, default=3,              #batch_size的大小
                        help="Specify batch size")

    parser.add_argument('--initial_epochs', type=int, default=50,          #初始化训练的epoch
                        help="Specify the number of epochs for initial training")

    parser.add_argument('--num_epochs', type=int, default=20,               #竞争训练的epoch
                        help="Specify the number of epochs for competitive search")

    parser.add_argument('--log_step', type=int, default=100,
                        help="Specify log step size for training")

    parser.add_argument('--learning_rate', type=float, default=1e-3,          #学习率设置
                        help="Learning rate")

    parser.add_argument('--data', type=str, default='',          #加载实验数据
                        help="Path to the dataset")

    parser.add_argument('--pop_size', type=int, default=18)                   #种群数量

    parser.add_argument('--code_length', type=int, default=10)                 #染色体长度

    parser.add_argument('--n_select', type=int, default=9)

    parser.add_argument('--time_steps', type=int, default=3)                   #时间步长

    parser.add_argument('--n_hidden', type=int, default=30)                    #隐含层的大小

    parser.add_argument('--n_output', type=int, default=1)                      #输出层的大小

    parser.add_argument('--max_grad_norm', type=float, default=1.0)             #该系数用于对梯度的裁剪，clipnorm的参数

    return parser.parse_args()


def main():
    args = parse_arguments()
    data, y_scaler = load_data_Duo(args.data)
    args.n_features = np.size(data, axis=-1)

    train, valid, test = data[0:1982, :], data[1982:2183, :] ,data[2183:2607, :]
    train_X, train_y = data_to_series_features(train, args.time_steps)
    valid_X, valid_y = data_to_series_features(valid, args.time_steps)
    test_X , test_y = data_to_series_features(test, args.time_steps)

    optimizer = optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.max_grad_norm)
    #优化算法Adam，在进行梯度下降算法过程中，梯度值可能会较大，通过控制clipnorm可以实现梯度的剪裁
    #使其2范数不超过clipnorm所设定的值

    best_model = make_model(args)   # 调用模型

    best_weight = [1.0] * args.time_steps   # 初始化best_weight
    best_model.compile(loss='mse', optimizer=optimizer)
    print("Initial training before competitive random search")
    best_model.fit(apply_weight(train_X, best_weight), train_y, epochs=args.initial_epochs,
                   validation_data=(apply_weight(valid_X, best_weight), valid_y),batch_size=args.batch_size,shuffle=True)
    print("\nInitial training is done. Start competitive random search.\n")

    pop, weights = initialize_weights(args.pop_size, args.time_steps, args.code_length)   #初始化种群和适应度
    key_to_rmse = {}           #定义一个空字典
    for iteration in range(args.iterations):
        for enum, (indiv, weight) in enumerate(zip(pop, weights)):         # zip() 数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            print('iteration: [%d/%d] indiv_no: [%d/%d]' % (iteration + 1, args.iterations, enum + 1, args.pop_size))
            key = individual_to_key(indiv)                                 #将个体转化为了字符串类型，用于后面的遍历
            if key not in key_to_rmse.keys():                              # .key() 函数用于返回字典所有类型
                model = make_model(args)
                model.compile(loss='mse', optimizer=optimizer)
                model.set_weights(best_model.get_weights())                #调用前面初始化训练的权重
                model.fit(apply_weight(train_X, weight), train_y, epochs=args.num_epochs,
                          validation_data=(apply_weight(valid_X, weight), valid_y),batch_size=args.batch_size,shuffle=True)
                pred_y = model.predict(apply_weight(valid_X, weight))
                inv_pred_y = y_scaler.inverse_transform(pred_y)
                inv_valid_y = y_scaler.inverse_transform(np.expand_dims(valid_y, axis=1))
                rmse1 = math.sqrt(mean_squared_error(inv_valid_y, inv_pred_y))
                mae1 = mean_absolute_error(inv_valid_y, inv_pred_y)
                r2_1 = r2_score(inv_valid_y, inv_pred_y)
                mape1 = MAPE(inv_valid_y, inv_pred_y)
                print("RMSE: %.4f, MAE: %.4f, mape : %.4f, r2: %.4f" % (rmse1, mae1,mape1,r2_1))
                if is_minimum(rmse1, key_to_rmse):                           #比较rmse，通过循环得出最小rmse
                    best_model.set_weights(model.get_weights())
                    best_weight = copy(weight)
                key_to_rmse[key] = rmse1

        pop_selected, fitness_selected = select(pop, args.n_select, key_to_rmse)
        pop = reconstruct_population(pop_selected, args.pop_size)
        weights = pop_to_weights(pop, args.time_steps, args.code_length)

    print('test evaluation:')
    pred_y = best_model.predict(apply_weight(test_X, best_weight))
    inv_pred_y = y_scaler.inverse_transform(pred_y)    #反归一化后的预测值
    print('inv_pred_y:',inv_pred_y.shape)

    inv_test_y = y_scaler.inverse_transform(np.expand_dims(test_y, axis=1))
    inv_test_y = np.reshape(inv_test_y,[421,1])

    rmse2 = math.sqrt(mean_squared_error(inv_pred_y, inv_test_y))
    mae2 = mean_absolute_error(inv_pred_y, inv_test_y)
    mape2 = MAPE(inv_pred_y, inv_test_y)
    r2_2 = r2_score(inv_pred_y, inv_test_y)
    print("RMSE: %.4f, MAE: %.4f, MAPE: %.4f, R2： %.4f" % (rmse2, mae2,mape2,r2_2))


if __name__ == '__main__':
    main()