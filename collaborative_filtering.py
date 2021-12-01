import numpy as np
from utils import preprocess, load_matrix


N = 10000  # 使用数据集大小


def similarity(train_mat):
    # 取测试集矩阵相乘等价于两两用户评分向量点乘求得分子
    num = np.matmul(train_mat, train_mat.transpose())
    # 取分子矩阵的对角线向量得到各用户评分向量的范数平方向量
    row_norm_vec = np.diagonal(num).reshape(-1, 1)
    # 将范数平方向量相乘并开方得到两两用户评分向量的范数相乘矩阵作为分母
    den = np.sqrt(np.matmul(row_norm_vec, row_norm_vec.transpose()))
    # 求得相似度矩阵
    sim = num / den

    return sim


def score_infer(sim, train_mat):
    # 取相似度矩阵和测试及矩阵相乘等价于用户相似度向量和用户评分向量点乘求得分子
    score = np.matmul(sim, train_mat)
    # 计算某用户与其他用户相似度之和作为分母，应注意减去自身与自身的相似度“1”
    sim_sum = (sim.sum(axis=0) - 1).reshape(-1, 1)
    # 求得评分矩阵
    score = score / sim_sum

    return score


def test(score, test_mat):
    # 计算测试集指示矩阵
    orient_test_mat = np.ones((N, N))
    orient_test_mat = np.where(test_mat > 0, orient_test_mat, test_mat)
    # 计算均方根误差
    rmse = np.sqrt((orient_test_mat * (score - test_mat)
                    ** 2).sum() / orient_test_mat.sum())

    return rmse


if __name__ == "__main__":
    # 读取训练集和测试及矩阵
    train_mat, test_mat = load_matrix(N)

    # 计算相似度矩阵
    sim = similarity(train_mat)

    # 计算评分矩阵
    score = score_infer(sim, train_mat)

    # 计算测试集均方根误差
    rmse = test(score, test_mat)

    print('Testing RMSE: {}'.format(rmse))
