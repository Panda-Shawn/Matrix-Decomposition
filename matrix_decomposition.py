import pandas as pd
import numpy as np
import os
import torch
from utils import preprocess, load_matrix_df
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

    orient_train_tensor = torch.ones((n, n)).to(device)
    orient_train_tensor = torch.where(
        train_tensor > 0, orient_train_tensor, train_tensor).to(device)

    # u_tensor = torch.rand(size=(N,k)).to(device)
    # v_tensor = torch.rand(size=(N,k)).to(device)
    # u_tensor = torch.normal(0.5, 0.1, size=(n, k)).to(device)
    # v_tensor = torch.normal(0.5, 0.1, size=(n, k)).to(device)
    u_tensor = torch.normal(0.1, 0.02, size=(n, k)).to(device)
    v_tensor = torch.normal(0.1, 0.02, size=(n, k)).to(device)

    u_tensor.requires_grad = True
    v_tensor.requires_grad = True
    optimizer = torch.optim.SGD([u_tensor, v_tensor], lr=lr)
    loss_train_rec = []
    rmse_train_rec = []

    for i in range(max_epoch):
        x_hat = u_tensor.matmul(v_tensor.t())

        loss_error = 0.5 * \
            torch.norm(orient_train_tensor *
                       (x_hat - train_tensor), p='fro') ** 2
        loss_u = lamda * torch.norm(u_tensor, p='fro') ** 2
        loss_v = lamda * torch.norm(v_tensor, p='fro') ** 2

        loss = loss_error + loss_u + loss_v
        loss_data = loss.detach().cpu().numpy()
        loss_train_rec.append(loss_data)

        rmse = (orient_train_tensor * (x_hat - train_tensor)
                ** 2).sum() / orient_train_tensor.sum()
        rmse = rmse.detach().cpu().numpy()
        rmse_train_rec.append(rmse)

        if i % 10 == 0:
            print('epoch: {}, loss: {}, RMSE: {}'.format(i, loss_data, rmse))
        if rmse < rmse_thres:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return u_tensor, v_tensor, loss_train_rec, rmse_train_rec


def test(u_tensor, v_tensor, test_tensor, n, device='cpu'):
    print("=" * 20 + "Start testing!" + "=" * 20)

    orient_test_tensor = torch.ones((n, n)).to(device)
    orient_test_tensor = torch.where(
        test_tensor > 0, orient_test_tensor, test_tensor).to(device)

    x_hat = u_tensor.matmul(v_tensor.t())
    rmse = (orient_test_tensor * (x_hat - test_tensor)
            ** 2).sum() / orient_test_tensor.sum()
    print('Testing RMSE: {}'.format(rmse))


def store(loss_train_rec, rmse_train_rec, k, lamda):
    store_root = 'results'
    if not os.path.exists(store_root):
        os.mkdir(store_root)

    loss_path = 'loss_rec_k{}_lamda{}.txt'.format(k, lamda)
    loss_path = os.path.join(store_root, loss_path)
    with open(loss_path, 'wt', encoding='utf-8') as f:
        for l in loss_train_rec:
            f.write(str(l) + '\n')

    rmse_path = 'RMSE_rec_k{}_lamda{}.txt'.format(k, lamda)
    rmse_path = os.path.join(store_root, rmse_path)
    with open(rmse_path, 'wt', encoding='utf-8') as f:
        for r in rmse_train_rec:
            f.write(str(r) + '\n')


if __name__ == "__main__":

    args = parse_args()
    n = args.n
    k = args.k
    lamda = args.lamda
    max_epoch = args.max_epoch
    lr = args.lr
    rmse_thres = args.rmse_thres
    use_cuda = args.cuda

    if not os.path.exists('train_df{}.csv'.format(n)) or not os.path.exists('test_df{}.csv'.format(n)):
        preprocess(n)

    train_df, test_df = load_matrix_df(n)
    train_mat = train_df.values
    test_mat = test_df.values

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

    train_tensor = torch.Tensor(train_mat).to(device)
    test_tensor = torch.Tensor(test_mat).to(device)

    u_tensor, v_tensor, loss_train_rec, rmse_train_rec = train(
        train_tensor, n, k, lamda, max_epoch, lr, rmse_thres, device)

    test(u_tensor, v_tensor, test_tensor, n, device)

    store(loss_train_rec, rmse_train_rec, k, lamda)
