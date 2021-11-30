import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    store_root = 'results'
    k_list = [20, 50]
    lamda_list = [1.0, 0.1, 0.01, 0.001]
    k, lamda = 50, 0.01
    loss_path = 'loss_rec_k{}_lamda{}.txt'.format(
        k, lamda)
    loss_path = os.path.join(store_root, loss_path)
    loss = []
    with open(loss_path, encoding='utf-8') as f:
        for line in f:
            loss.append(float(line.strip()))

    loss = np.array(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss, label='$\lambda=${}'.format(lamda))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('loss of training data when $k=${}'.format(k))
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(k_list)):

        loss = [[] for i in range(len(lamda_list))]
        for j in range(len(lamda_list)):

            loss_path = 'loss_rec_k{}_lamda{}.txt'.format(
                k_list[i], lamda_list[j])
            loss_path = os.path.join(store_root, loss_path)
            loss = []
            with open(loss_path, encoding='utf-8') as f:
                for line in f:
                    loss.append(float(line.strip()))

            loss = np.array(loss)
            ax.plot(loss, label='$k=${}, $\lambda=${}'.format(
                k_list[i], lamda_list[j]))

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    # ax.set_title('loss of training data when $k=${}'.format(k_list[i]))
    ax.set_title('loss of training data')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(k_list)):

        for j in range(len(lamda_list)):
            rmse_path = 'RMSE_rec_k{}_lamda{}.txt'.format(
                k_list[i], lamda_list[j])
            rmse_path = os.path.join(store_root, rmse_path)
            rmse = []
            with open(rmse_path, encoding='utf-8') as f:
                for line in f:
                    rmse.append(float(line.strip()))

            rmse = np.array(rmse)
            ax.plot(rmse, label='$k=${}, $\lambda=${}'.format(
                k_list[i], lamda_list[j]))

    ax.set_xlabel('epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE of training data')
    ax.legend()
    plt.show()
