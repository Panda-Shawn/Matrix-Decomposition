import pandas as pd
import numpy as np
import os
import torch
from utils import preprocess, load_matrix
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix decomposition")
    parser.add_argument("-n", type=int, default=10000,
                        help="The amount of data used")
    parser.add_argument(
        "-k", type=int, default=50, help="The amount of eigenvectors used")
    parser.add_argument("-l", "--lamda", type=float, default=0.01,
                        help="The factor of regulization")
    parser.add_argument("-me", "--max_epoch", type=int, default=300,
                        help="The maximum of epochs")
    parser.add_argument("-lr", type=float, default=0.00001,
                        help="The learning rate of training")
    parser.add_argument("-r", "--rmse_thres", type=float,
                        default=-1.0, help="The threshold of RMSE")
    parser.add_argument("-c", "--cuda", type=bool,
                        default=False, help="Whether to cuda")

    args = parser.parse_args()
    return args


def train(train_tensor,
          n,
          k,
          lamda,
          max_epoch,
          lr,
          rmse_thres,
          device='cpu'):
    print("=" * 20 + "Start training!" + "=" * 20)

    # 计算训练集指示矩阵张量
    orient_train_tensor = torch.ones((n, n)).to(device)
    orient_train_tensor = torch.where(
        train_tensor > 0, orient_train_tensor, train_tensor).to(device)

    # 初始化u，v张量
    # u_tensor = torch.rand(size=(N,k)).to(device)
    # v_tensor = torch.rand(size=(N,k)).to(device)
    # u_tensor = torch.normal(0.5, 0.1, size=(n, k)).to(device)
    # v_tensor = torch.normal(0.5, 0.1, size=(n, k)).to(device)
    u_tensor = torch.normal(0.1, 0.02, size=(n, k)).to(device)
    v_tensor = torch.normal(0.1, 0.02, size=(n, k)).to(device)

    # 利用torch进行梯度下降计算，要将u，v作为计算图中的叶子节点计算梯度
    u_tensor.requires_grad = True
    v_tensor.requires_grad = True

    # 使用SGD优化器
    optimizer = torch.optim.SGD([u_tensor, v_tensor], lr=lr)

    # 记录loss和RMSE
    loss_train_rec = []
    rmse_train_rec = []

    for i in range(max_epoch):
        # 计算得到x_hat
        x_hat = u_tensor.matmul(v_tensor.t())

        # 计算uv乘积逼近x的误差
        loss_error = 0.5 * \
            torch.norm(orient_train_tensor *
                       (x_hat - train_tensor), p='fro') ** 2
        # 计算u正则项
        loss_u = lamda * torch.norm(u_tensor, p='fro') ** 2
        # 计算v正则项
        loss_v = lamda * torch.norm(v_tensor, p='fro') ** 2

        # 三项求和计算总loss
        loss = loss_error + loss_u + loss_v

        # 记录loss
        loss_data = loss.detach().cpu().numpy()
        loss_train_rec.append(loss_data)

        # 计算RMSE
        rmse = torch.sqrt((orient_train_tensor * (x_hat - train_tensor)
                           ** 2).sum() / orient_train_tensor.sum())

        # 记录RMSE
        rmse = rmse.detach().cpu().numpy()
        rmse_train_rec.append(rmse)

        # 每隔10个epoch打印loss和RMSE信息
        if i % 10 == 0:
            print('epoch: {}, loss: {}, RMSE: {}'.format(i, loss_data, rmse))

        # 若设置了RMSE阈值则小于阈值时结束训练
        if rmse < rmse_thres:
            break

        # 计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 更新u，v张量
        optimizer.step()
    return u_tensor, v_tensor, loss_train_rec, rmse_train_rec


def test(u_tensor, v_tensor, test_tensor, n, device='cpu'):
    print("=" * 20 + "Start testing!" + "=" * 20)

    # 计算测试集指示矩阵张量
    orient_test_tensor = torch.ones((n, n)).to(device)
    orient_test_tensor = torch.where(
        test_tensor > 0, orient_test_tensor, test_tensor).to(device)

    # 计算x_hat
    x_hat = u_tensor.matmul(v_tensor.t())

    # 计算RMSE
    rmse = (orient_test_tensor * (x_hat - test_tensor)
            ** 2).sum() / orient_test_tensor.sum()
    print('Testing RMSE: {}'.format(rmse))


def store(loss_train_rec, rmse_train_rec, k, lamda):
    store_root = 'results'
    if not os.path.exists(store_root):
        os.mkdir(store_root)

    # 存储loss数据用于画图
    loss_path = 'loss_rec_k{}_lamda{}.txt'.format(k, lamda)
    loss_path = os.path.join(store_root, loss_path)
    with open(loss_path, 'wt', encoding='utf-8') as f:
        for l in loss_train_rec:
            f.write(str(l) + '\n')

    # 存储RMSE数据用于画图
    rmse_path = 'RMSE_rec_k{}_lamda{}.txt'.format(k, lamda)
    rmse_path = os.path.join(store_root, rmse_path)
    with open(rmse_path, 'wt', encoding='utf-8') as f:
        for r in rmse_train_rec:
            f.write(str(r) + '\n')


if __name__ == "__main__":

    # 读取相关参数
    args = parse_args()
    n = args.n
    k = args.k
    lamda = args.lamda
    max_epoch = args.max_epoch
    lr = args.lr
    rmse_thres = args.rmse_thres
    use_cuda = args.cuda

    # 判断是否有已经预处理好的训练集和测试集矩阵，若没有，重新预处理
    if not os.path.exists('train_mat{}.npy'.format(n)) or not os.path.exists('test_mat{}.npy'.format(n)):
        preprocess(n)

    # 读取训练集和测试集矩阵
    train_mat, test_mat = load_matrix(n)

    # 判断是否使用cuda
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

    # 将矩阵张量化
    train_tensor = torch.Tensor(train_mat).to(device)
    test_tensor = torch.Tensor(test_mat).to(device)

    # 进行训练
    u_tensor, v_tensor, loss_train_rec, rmse_train_rec = train(
        train_tensor, n, k, lamda, max_epoch, lr, rmse_thres, device)

    # 进行测试
    test(u_tensor, v_tensor, test_tensor, n, device)

    # 存储loss和RMSE数据
    store(loss_train_rec, rmse_train_rec, k, lamda)
