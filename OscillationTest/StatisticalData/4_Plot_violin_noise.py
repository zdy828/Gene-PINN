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
    OSC_01 = torch.load('StatisticalData/errors_noise_0.01.pt')
    OSC_05 = torch.load('StatisticalData/errors_noise_0.05.pt')
    OSC_1 = torch.load('StatisticalData/errors_noise_0.1.pt')
    OSC_01 = torch.from_numpy(OSC_01)
    OSC_05 = torch.from_numpy(OSC_05)
    OSC_1 = torch.from_numpy(OSC_1)

    # 将所有数组沿一个新的轴堆叠，得到形状 (100, 5, 4)
    stacked = np.stack([OSC_01, OSC_05, OSC_1], axis=2)

    # 按列索引提取，得到 5 个 (100, 3) 的数组
    OSC_param = [stacked[:, i, :] for i in range(5)]

    # 小提琴图
    result_label = ['0.01', '0.05', '0.1']
    param_labels = ['x_train', 'x_test', 'r', 'K', 'd']

    for i in range(len(param_labels)):
        plt.figure(figsize=(8, 8))
        # plt.subplot(221)
        plt.grid(True)
        ax = sns.violinplot(OSC_param[i], palette="Set2")
        plt.ylim(0, 1)
        plt.title(param_labels[i], fontsize=28)
        plt.xticks([0, 1, 2], result_label)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=24)
        plt.tick_params(axis='both', labelsize=24)  # 调整刻度标签字体大小（数字）
        plt.savefig('4_Plot_violin_noise/'+param_labels[i]+'.png', bbox_inches='tight', dpi=600)
        # plt.show()
        plt.close()  # 关闭图形，释放内存





