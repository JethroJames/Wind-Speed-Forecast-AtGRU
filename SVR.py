import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from vmdpy import VMD
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import attention
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))*100


class SVR():
    def __init__(self,X_train,Y_train,X_test,Y_test,scaled_tool,final_Xtest,final_Ytest):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.scaled_tool = scaled_tool
        self.x_test = final_Xtest
        self.y_test = final_Ytest
    def run(self):
        #搭建模型
        model = svm.SVR(kernel='linear')
        model.fit(self.X_train, self.Y_train)
        Y_pre = model.predict(self.X_test)

        Y_pre = Y_pre.reshape(self.X_test.shape[0], 1)
        Y_test = self.Y_test.reshape(self.X_test.shape[0], 1)
        Y_pre = self.scaled_tool.inverse_transform(Y_pre)
        Y_test = self.scaled_tool.inverse_transform(Y_test)

        '''第二次预测'''
        predict2 = model.predict(self.x_test)
        predict2 = predict2.reshape(-1,1)
        pred_2 = self.scaled_tool.inverse_transform(predict2)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)
        test_2 = self.scaled_tool.inverse_transform(self.y_test)
        # 计算初次预测评价指标
        print('first MAPE : ', mape(Y_test, Y_pre))
        print('first MAE : ', mae(Y_test, Y_pre))
        print('first R2 : ', r2(Y_test, Y_pre))
        print('first RMSE : ', np.sqrt(mse(Y_test, Y_pre)))
        metrics0 = []
        metrics0.append([mape(Y_test, Y_pre), mae(Y_test, Y_pre), r2(Y_test, Y_pre), np.sqrt(mse(Y_test, Y_pre))])
        metrics0_ = pd.DataFrame(metrics0, columns=['MAPE', ' MAE', 'R2', 'RMSE'])
        metrics0_.to_excel(r"SVR评价数据\SVR 初预测效果评价1.xlsx")
        #
        # 计算第二次预测评价指标
        print('second MAPE : ', mape(test_2, pred_2))
        print('second MAE : ', mae(test_2, pred_2))
        print('second R2 : ', r2(test_2, pred_2))
        print('second RMSE : ', np.sqrt(mse(test_2, pred_2)))
        metrics1 = []
        metrics1.append([mape(test_2, pred_2), mae(test_2, pred_2), r2(test_2, pred_2), np.sqrt(mse(test_2, pred_2))])
        metrics1_ = pd.DataFrame(metrics1, columns=['MAPE', ' MAE', 'R2', 'RMSE'])
        metrics1_.to_excel(r"SVR评价数据\SVR 初预测效果评价2.xlsx")
        # 绘制结果图1
        plt.figure(1)
        plt.plot(Y_pre, color='red', label='Predicted wind speed')
        plt.plot(Y_test, color='yellow', label='True value of wind speed')
        plt.title("wind speed")
        plt.xlabel("Time")
        plt.ylabel("Wind speed value")
        plt.legend()
        plt.savefig(r"SVR评价数据\SVR 预测图1")
        plt.show()
        # 绘制结果图2
        plt.figure(2)
        plt.plot(pred_2, color='red', label='Predicted wind speed')
        plt.plot(test_2, color='yellow', label='True value of wind speed')
        plt.title("wind speed")
        plt.xlabel("Time")
        plt.ylabel("Wind speed value")
        plt.legend()
        plt.savefig(r"SVR评价数据\SVR 预测图2")
        plt.show()
        # 保存数据与模型
        total_error2 = np.append(Y_pre - Y_test, pred_2 - test_2)
        np.savetxt('SVR预测数据\SVR Preliminary label.csv', Y_test, delimiter=',')
        np.savetxt('SVR预测数据\SVR Preliminary forecast.csv', Y_pre, delimiter=',')
        np.savetxt('SVR预测数据\SVR Predict_error.csv', Y_pre - Y_test, delimiter=',')
        np.savetxt('SVR预测数据\SVR Final forecast.csv', pred_2, delimiter=',')
        np.savetxt('SVR预测数据\SVR Fianl_error.csv', pred_2 - test_2, delimiter=',')
        np.savetxt('SVR预测数据\SVR Final label.csv', test_2, delimiter=',')
        np.savetxt('SVR预测数据\TotalError.csv', total_error2, delimiter=',')
 # 搭建误差序列分解类(内部类)
    '''1.read error data set'''
    '''2.process error sequence'''
    class ErrorProcessed():
        def __init__(self, name1, name2):
            self.alpha = 5000  # moderate bandwidth constraint
            self.tau = 0.  # noise-tolerance (no strict fidelity enforcement)
            self.K = 10  # 10 modes
            self.DC = 0  # no DC part imposed
            self.init = 1  # initialize omegas uniformly
            self.tol = 1e-7
            self.name_ = name1  # 评价指标对应模型的标签
            self.name = name2  # 预测模型对应标签
            self.f = pd.read_csv(f'{self.name}\TotalError.csv', header=None)
            self.u, self.u_hat, self.omega = VMD(self.f.values, self.alpha, self.tau, self.K, self.DC, self.init,
                                                 self.tol)

        """  
        alpha、tau、K、DC、init、tol 六个输入参数的无严格要求； 
        alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍； 
        tau 噪声容限 ；
        K 分解模态（IMF）个数； 
        DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1； 
        init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数； 
        tol 控制误差大小常量，决定精度与迭代次数
        """

        def run1(self):
            plt.figure(1)
            for i in range(self.K):
                plt.subplot(self.K, 1, i + 1)
                plt.plot(self.u[i, :], linewidth=0.2, c='r')
                plt.ylabel('IMF{}'.format(i + 1))
            plt.savefig(f'{self.name_}\Procedure1 of VMD ')
            # 中心模态
            plt.figure(2)
            for i in range(self.K):
                plt.subplot(self.K, 1, i + 1)
                plt.plot(abs(fft(self.u[i, :])))
                plt.ylabel('IMF{}'.format(i + 1))
            plt.savefig(f'{self.name_}\Procedure2 of VMD ')
            fig1 = plt.figure(4)
            plt.plot(self.f.values)
            plt.plot(self.u, c='r')
            fig1.suptitle('Result Of VMD')
            # 保存子序列数据到文件中
            np.savetxt(f'{self.name}\imfs.csv', self.u.T, delimiter=',')
            plt.savefig(f'{self.name_}\Result Of VMD')
            plt.show()

        # 3.build Error-Compensation-model for fitting test sequence
        def run2(self):
            metrics_ = []
            sum = 0
            '''对每个子模态进行训练并预测'''
            datasets = pd.read_csv(f'{self.name}\imfs.csv', header=None)
            for n in range(0, 10):
                # dataset = datasets.iloc[:,[5]].values
                dataset = datasets.iloc[:, n].values.reshape(-1, 1)
                # 归一化
                scaled_tool = MinMaxScaler(feature_range=[0, 1])
                data_scaled = scaled_tool.fit_transform(dataset)
                # 切片
                step_size = 60
                data_seg = np.zeros((len(data_scaled) - step_size, step_size))
                for i in range(len(data_scaled) - step_size):
                    data_seg[i, :] = data_scaled[i:step_size + i, 0]
                data_label = data_scaled[step_size:, 0]
                # 划分数据集
                test_number = 400
                X_train = data_seg[:-test_number]
                Y_train = data_label[:-test_number]
                X_test = data_seg[-test_number:]
                Y_test = data_label[-test_number:]
                # x, y = X_train.shape[0], X_train.shape[1]
                # 张量转化
                # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # 搭建模型
                model = svm.SVR(kernel='linear')
                model.fit(X_train, Y_train)
                # 每个子模态模型进行预测并收集预测数据
                predict = model.predict(X_test)
                predict = predict.reshape(-1, 1)
                pred_ = scaled_tool.inverse_transform(predict)  # 将标准化后的数据转化为原始数据
                Y_test = Y_test.reshape(Y_test.shape[0], 1)
                test_ = scaled_tool.inverse_transform(Y_test)  # 将标签数据也转化为原始数据
                print(f'IMF{n + 1}MAPE : ', mape(test_, pred_))
                print(f'IMF{n + 1}MAE : ', mae(test_, pred_))
                print(f'IMF{n + 1}R2 : ', r2(test_, pred_))
                print(f'IMF{n + 1}RMSE : ', np.sqrt(mse(test_, pred_)))
                sum += pred_
                metrics_.append([mape(test_, pred_), mae(test_, pred_), r2(test_, pred_), np.sqrt(mse(test_, pred_))])
            data = pd.DataFrame(metrics_, columns=['MAPE', 'MAE', 'R2', 'RMSE'],
                                index=[f'IMF{i}' for i in range(1, 11)])
            data.to_excel(rf"{self.name_}\vmd分解结果.xlsx")
            df = pd.DataFrame(sum)
            df.to_excel(rf"{self.name}\vmd分解后的误差补偿序列.xlsx")
            return sum

        def run3(self, error_sequence):
            '''5.最后评价'''
            # 读入评测的标签数据以及预测数据
            label = pd.read_csv(f'{self.name}\SVR Final label.csv', header=None)
            pre = pd.read_csv(f'{self.name}\SVR Final forecast.csv', header=None)
            # dataset = datasets.iloc[:,[5]].values
            label_ = label.iloc[:, 0].values.reshape(-1, 1)
            pre_ = pre.iloc[:, 0].values.reshape(-1, 1)
            pre_ -= error_sequence
            # 评价最终预测效果
            print('final MAPE : ', mape(label_, pre_))
            print('final MAE : ', mae(label_, pre_))
            print('final R2 : ', r2(label_, pre_))
            print('final RMSE : ', np.sqrt(mse(label_, pre_)))
            metrics = []
            metrics.append([mape(label_, pre_), mae(label_, pre_), r2(label_, pre_), np.sqrt(mse(label_, pre_))])
            metrics2 = pd.DataFrame(metrics, columns=['MAPE', 'MAE', 'R2', 'RMSE'])
            metrics2.to_excel(f"{self.name_}\ECSVR预测效果评价.xlsx")
            plt.plot(pre_, color='red', label='Predicted wind speed')
            plt.plot(label_, color='yellow', label='True value of wind speed')
            plt.title("wind speed")
            plt.xlabel("Time")
            plt.ylabel("Wind speed value")
            plt.legend()
            plt.savefig(f"{self.name_}\ECSVR预测图")
            plt.show()

