import numpy as np
from utils import preprocess, load_matrix_df


N = 10000


def similarity(train_mat):
    num = np.matmul(train_mat, train_mat.transpose())
    row_norm_vec = np.diagonal(num).reshape(-1, 1)
    den = np.sqrt(np.matmul(row_norm_vec, row_norm_vec.transpose()))
    sim = num / den

    return sim


def score_infer(sim, train_mat):
    score = np.matmul(sim, train_mat)
    sim_sum = (sim.sum(axis=0) - 1).reshape(-1,
                                            1).repeat(len(sim), axis=1)
    score = score / sim_sum

    return score


def test(score, test_mat):
    orient_test_mat = np.ones((N, N))
    orient_test_mat = np.where(test_mat > 0, orient_test_mat, test_mat)
    rmse = (orient_test_mat * (score - test_mat)
            ** 2).sum() / orient_test_mat.sum()

    return rmse


if __name__ == "__main__":
    train_df, test_df = load_matrix_df(N)
    train_mat = train_df.values
    test_mat = test_df.values

    sim = similarity(train_mat)
    score = score_infer(sim, train_mat)

    rmse = test(score, test_mat)

    print('Testing RMSE: {}'.format(rmse))
