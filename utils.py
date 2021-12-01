import pandas as pd
import numpy as np
from collections import defaultdict


def preprocess(n):
    # 初始化矩阵
    train_mat = np.zeros((n, n))
    test_mat = np.zeros((n, n))

    # 由于用户id无法直接作为索引使用，用字典对id进行转换，查找速度快
    user = defaultdict(int)
    with open('data/users.txt', encoding='utf-8') as f:
        num_line = 0
        for line in f:
            user[line.strip()] = num_line
            num_line += 1
            if num_line >= n:
                break

    # 读取训练集数据，向训练集矩阵中填值
    with open('data/netflix_train.txt', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            train_mat[user[data[0]]][int(data[1])-1] = int(data[2])

    # 读取测试集数据，向测试集矩阵中填值
    with open('data/netflix_test.txt', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            test_mat[user[data[0]]][int(data[1])-1] = int(data[2])

    # 存储训练集、测试集矩阵
    np.save('train_mat{}.npy'.format(n), train_mat)
    np.save('test_mat{}.npy'.format(n), test_mat)


def load_matrix(n):
    # 读取训练集、测试集矩阵
    train_mat = np.load('train_mat{}.npy'.format(n))
    test_mat = np.load('test_mat{}.npy'.format(n))
    return train_mat, test_mat
