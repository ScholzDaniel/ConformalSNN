import numpy as np

from utils import one_hot_to_sparse, write_data


def get_coverage_list(pred_hist, labels):
    sum_tot = []
    labels_sparse = one_hot_to_sparse(labels)
    for i, preds in enumerate(pred_hist):
        if labels_sparse[i] in preds:
            sum_tot.append(1)
        else:
            sum_tot.append(0)
    return sum_tot


def get_coverage(pred_hist, labels):
    covs = get_coverage_list(pred_hist, labels)
    cov = np.sum(covs)/len(covs)
    return cov


def acc_vs_len(cp):
    if cp.test_labels is None:
        raise AttributeError("Pass labels to the CP predict method.")
    y_true = np.argmax(cp.test_labels, axis=1)
    y_hat = np.argmax(cp.test_scores, axis=1)
    len_hist = np.array(cp.len_hist)
    smooth_len_hist = np.array(cp.smooth_len_hist)
    smooth_c = smooth_len_hist[y_true == y_hat]
    smooth_w = smooth_len_hist[y_true != y_hat]
    c = len_hist[y_true == y_hat]
    w = len_hist[y_true != y_hat]

    result_dict = {}
    names = ['sc', 'sw', 'c', 'w']
    vas = [smooth_c, smooth_w, c, w]

    for name, var in zip(names, vas):
        result_dict[f'median_{name}'] = np.round(np.median(var), 2)
        result_dict[f'mean_{name}'] = np.round(np.mean(var), 2)
        result_dict[f'std_{name}'] = np.round(np.std(var), 2)
    return result_dict


def mean_len_against_frames(cp, hist):
    """
    :param cp: CP instance
    :type cp: cp_uncertainty.MovAvgConformalPrediction
    :param hist: CP History
    :type hist: list
    :return:
    """
    h = [c[0] for c in cp.pred_consist_hist]
    match = np.equal(np.array(h), np.argmax(cp.test_labels, axis=1)).astype(
        np.uint8)
    match = match.reshape((-1, cp.frames))
    acc = np.mean(match, axis=0)

    uc = np.array(hist).reshape((-1, cp.frames))
    mean_std_c, mean_std_w = reduce_wrong_correct(match, uc, std=False)
    return acc, mean_std_c, mean_std_w


def reduce_wrong_correct(match, uc, std=False):
    mean_uc_c = np.add.reduce(uc * match, axis=0,
                              keepdims=True) / np.add.reduce(match + 1e-20,
                                                             axis=0)
    mean_uc_w = np.add.reduce(uc * (1 - match), axis=0,
                              keepdims=True) / np.add.reduce(
        (1 - match) + 1e-20, axis=0)
    std_uc_c = np.std(uc * match, axis=0,
                      keepdims=True)
    std_uc_w = np.std(uc * (1 - match), axis=0,
                      keepdims=True)
    if std:
        return np.vstack((mean_uc_c, std_uc_c)),\
               np.vstack((mean_uc_w, std_uc_w))
    else:
        return mean_uc_c, mean_uc_w


def mean_softmax_against_frames(y_hat, y_true, frames):
    match = np.equal(np.argmax(y_hat, axis=-1), y_true).astype(np.uint8)
    match = match.reshape((-1, frames))
    score_max = np.max(y_hat, axis=-1)
    score_max = score_max.reshape((-1, frames))
    uc = 1 - score_max
    return match, uc


def save_hist_and_cov_by_slice(cp, s):
    x = list(range(s.stop - s.start))
    coverage = get_coverage_list(cp.pred_consist_hist, cp.test_labels)
    y_cov = coverage[s]
    y_slen = cp.smooth_len_hist[s]
    y_len = cp.len_hist[s]
    write_data(x, y_cov, f'cov_{s.start}')
    write_data(x, y_len, f'len_{s.start}')
    write_data(x, y_slen, f'smooth_comb_len_{s.start}')
