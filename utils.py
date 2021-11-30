import pandas as pd
import numpy as np


def preprocess(n):
    # 初始化矩阵
    train_init_mat = np.zeros((n, n))
    test_init_mat = np.zeros((n, n))

    movie = []
    with open('data/movie_titles.txt', encoding='ISO-8859-1') as f:
        num_line = 0
        for line in f:
            line = line.strip().split(',')
            movie.append(line[0])
            num_line += 1
            if num_line >= n:
                break

    user = []
    with open('data/users.txt', encoding='utf-8') as f:
        num_line = 0
        for line in f:
            user.append(line.strip())
            num_line += 1
            if num_line >= n:
                break

    # 初始化10000*10000（用户*电影）表格
    score_df_train = pd.DataFrame(train_init_mat, index=user, columns=movie)
    score_df_test = pd.DataFrame(test_init_mat, index=user, columns=movie)

    with open('data/netflix_train.txt', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            try:
                score_df_train[data[1]][data[0]] = int(data[2])
            except:
                pass

    with open('data/netflix_test.txt', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            try:
                score_df_test[data[1]][data[0]] = int(data[2])
            except:
                pass

    score_df_train.to_csv('train_df{}.csv'.format(n))
    score_df_test.to_csv('test_df{}.csv'.format(n))


def load_matrix_df(n):
    train_df = pd.read_csv('train_df{}.csv'.format(n), index_col=0)
    test_df = pd.read_csv('test_df{}.csv'.format(n), index_col=0)
    return train_df, test_df
