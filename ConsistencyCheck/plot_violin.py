# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Libraries and Dependencies
# 统计数据
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import datetime
import os
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Configurations
if __name__ == '__main__':
    T10 = torch.load('TrainingResults/StatisticalData/errors_T10.pt')
    T100 = torch.load('TrainingResults/StatisticalData/errors_T100.pt')
    T1000 = torch.load('TrainingResults/StatisticalData/errors_T1000.pt')
    T10 = torch.from_numpy(T10)
    T100 = torch.from_numpy(T100)
    T1000 = torch.from_numpy(T1000)
    x = np.arange(0,100).T
    # x = np.arange(0, 89).T
    plt.figure(figsize=(16, 4))
    labels = ['x_train','x_test', 'r','K','d']
    plt.subplot(131)
    plt.grid(True)
    sns.violinplot(T10, palette="Set3")
    plt.title('T10*G10')
    plt.xticks([0, 1, 2, 3, 4], labels)


    plt.subplot(132)
    plt.grid(True)
    sns.violinplot(T100, palette="Set3")
    plt.title('T100')
    plt.xticks([0, 1, 2, 3, 4], labels)

    plt.subplot(133)
    plt.grid(True)
    sns.violinplot(T1000, palette="Set3")
    plt.title('T1000')
    plt.xticks([0, 1, 2, 3, 4], labels)
    plt.savefig('Results.png', bbox_inches='tight', dpi=600)
    plt.show()

    # 箱线图
    # labels = 'a', 'K', 'd', 'x'
    # plt.subplot(223)
    # plt.grid(True)
    # plt.boxplot(T10.T, labels=labels)
    # plt.title('T10*G10')

    # plt.subplot(224)
    # plt.grid(True)
    # plt.boxplot(T100.T, labels=labels)
    # plt.title('T100')
    # plt.savefig('Results.png', bbox_inches='tight', dpi=600)
    # plt.show()






