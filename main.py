# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import attention
from scipy.fftpack import fft
from SVR import SVR
from BPNN import BPNN
from AtGRU import  AtGRU
from RNN import RNN
from GRU import GRU
from vmdpy import VMD
from AtGRUndGRU import AtGRUndGRU
def initdir():
    # 创建文件夹
    # names = ['AtGRU评价数据','AtGRU预测数据','BPNN预测数据','BPNN评价数据','SVR评价数据','SVR预测数据',
    #          'GRU评价数据','GRU预测数据','RNN评价数据','RNN预测数据']
    #names = ['AtGRU评价数据', 'AtGRU预测数据','GRU评价数据','GRU预测数据','AtGRUndGRU评价数据','AtGRUndGRU预测数据']
    names =['各模型评价数据','各模型预测数据']
    for i in range(len(names)):
        path = f'./{names[i]}'#在指定路径下创建文件夹用于承装数据
        if not os.path.exists(path):
            os.makedirs(path)

def get_names():
    Metrics = ['各模型评价数据']
    predicts = ['各模型预测数据']
    return Metrics,predicts

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))*100

def DataDevision():
    #导入数据
    datasets = pd.read_csv('SSA_data 1h.csv',header=None)
    # dataset = datasets.iloc[:,[5]].values
    dataset = datasets.iloc[:,0].values.reshape(-1,1)
    #归一化
    scaled_tool = MinMaxScaler(feature_range=[0, 1])
    data_scaled = scaled_tool.fit_transform(dataset)
    #切片
    step_size = 60
    data_seg = np.zeros((len(data_scaled)-step_size-1,step_size))
    for i in range(len(data_scaled)-step_size-1):
        data_seg[i,:] = data_scaled[i:step_size+i,0]
    data_label = data_scaled[step_size+1:,0]
    #划分数据集
    test_number2 = 400
    test_number1 = 700#(3191-400)*800/2135
    #print(len(data_seg)) #3191
    X = data_seg[:-test_number2]   #用于初训练的总样本
    Y = data_label[:-test_number2] #用于初训练的总标签
    x_train = X[:-test_number1]
    y_train = Y[:-test_number1]
    x_test  = X[-test_number1:]
    y_test  = Y[-test_number1:]
    '''用于最终测试的数据'''
    X_test = data_seg[-test_number2:]    #用于最终效果测试的全样本数据
    Y_test = data_label[-test_number2:]  #用于最终效果测试的全标签数据
    return x_train,y_train,x_test,y_test,X_test,Y_test,scaled_tool

if __name__ == '__main__':
    initdir()#创建好文件夹放入数据
    '''1.导入数据，划分数据集'''
    x_train,y_train,x_test,y_test,X_test,Y_test,scaled_tool = DataDevision()
    '''2.初次预测'''
    #获取模型对象
    #本次实验主要是比较不同模型训练同一误差序列的效果，用的是AtGRU初预测后的误差序列以及VMD序列分解方法
    process= AtGRUndGRU(x_train,y_train,x_test,y_test,scaled_tool,X_test,Y_test)#创建一个实验进程对象
    #开始初预测
    process.run()
    '''3.vmd分解误差序列'''
    Metrics,predicts = get_names()#names1对应评价指标数据，names2对应预测数据
    #获取误差序列分解系统对象
    vmd = process.ErrorProcessed(Metrics[0], predicts[0])
    #获取vmd处理的初步误差序列（提取误差序列的主要特征）
    vmd.runvmd()
    #对误差序列分解对象进行相应训练,并获取训练后的误差补偿序列
    error_sequence1 = vmd.runGRU()
    error_sequence2 = vmd.runAtGRU()
    error_sequence3 = vmd.runRNN()
    error_sequence4 = vmd.runBPNN()
    error_sequence5 = vmd.runSVR()
    #完成最终预测以及相应评价
    vmd.run3(error_sequence1,'GRU')
    vmd.run3(error_sequence2,'AtGRU')
    vmd.run3(error_sequence3,'RNN')
    vmd.run3(error_sequence4,'BPNN')
    vmd.run3(error_sequence5,'SVR')
    print("处理完毕！")