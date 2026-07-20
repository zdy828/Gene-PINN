# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Libraries and Dependencies
## 主要对没有噪声的数据，加上一点外部噪声，用于训练

import sys
sys.path.insert(0, '../Utilities/')

import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import datetime
import os
from matplotlib.ticker import MaxNLocator
np.random.seed(42)


warnings.filterwarnings('ignore')

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    # 导入原来没有噪声的数据
    x_no_noise = torch.load('ParameterGeneration/no_noise/fit_x.pt')  # [N1,5]
    N1 = 91  # 数据一共91个时间点
    t = np.linspace(0.05, 0.95, N1)
    time = t.reshape((N1, 1))
    n_genes = 4

    # 导入4组加入外部随机噪声的数据
    deltas = [0.01, 0.05, 0.1]  # [0.005, 0.01, 0.05, 0.1]
    noisy_datasets = {}

    for delta in deltas:
        file_path = f'ParameterGeneration/noise_x/noisy_data_delta_{delta}.npy'
        noisy_datasets[delta] = np.load(file_path)
        print(f"已导入: noisy_data_delta_{delta}.npy, 形状: {noisy_datasets[delta].shape}")


    # 画图展示没有噪声的数据和加上噪声的数据
    gene_expr = x_no_noise
    gene_name = ['hb', 'Kr', 'Kni', 'gt']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 只用前4个 # 蓝，橙，绿，红，紫
    colors = ['#ff7f0e','#9467bd', '#2ca02c']

    for gene_idx in range(n_genes):
        # 创建独立的 Figure
        plt.figure(figsize=(10, 8))

        # 绘制无噪声数据（黑色灰色粗实线）
        plt.plot(time, gene_expr[:, gene_idx],
                 color='lightgray', linewidth=8, label='No noise', alpha=1)

        # 绘制不同噪声水平的数据（虚线，不同颜色）
        for i, delta in enumerate(deltas):
            plt.plot(time, noisy_datasets[delta][:, gene_idx],
                     color=colors[i], linewidth=4, alpha=0.8,
                     label=f'Noise δ={delta}M', linestyle='--')

        # 标题、轴标签、网格
        plt.title(gene_name[gene_idx], fontsize=28, fontweight='bold')
        plt.tick_params(axis='both', labelsize=24)  # 调整刻度标签字体大小（数字）
        # plt.xlabel('Time Points', fontsize=10)
        # plt.ylabel('Expression Level', fontsize=10)
        plt.grid(True, alpha=0.3)

        # 只在最后一个基因（索引3）添加图例
        if gene_idx == n_genes - 1:
            plt.legend(loc='best', fontsize=22)

        # 保存为独立图片
        plt.savefig(f'ParameterGeneration/noise_x/gene_{gene_name[gene_idx]}_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭当前图形，释放内存

    print("所有基因图已单独保存，图例放在第4个图上。")



