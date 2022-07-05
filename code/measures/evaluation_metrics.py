import numpy as np

def get_accuracy(predict, target):
    p = np.array(predict)
    t = np.array(target)
    return np.average(t == p)


def get_true_positive_count(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return np.count_nonzero(target[predict == 1] == 1)


def get_false_positive_count(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return np.count_nonzero(target[predict == 1] == 0)


def get_true_negative_count(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return np.count_nonzero(target[predict == 0] == 0)


def get_false_negative_count(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return np.count_nonzero(target[predict == 0] == 1)


def get_recall(predict, target):
    t_p = get_true_positive_count(predict, target)
    f_n = get_false_negative_count(predict, target)
    if t_p == 0:
        return 0.0
    return t_p / (t_p + f_n)


def get_precision(predict, target):
    t_p = get_true_positive_count(predict, target)
    f_p = get_false_positive_count(predict, target)
    if t_p == 0:
        return 0.0
    return t_p / (t_p + f_p)


def get_f_score(predict, target, betha = 1):
    precision = get_precision(predict, target)
    recall = get_recall(predict, target)
    if precision == 0 or recall == 0:
        return 0.0
    return (1 + betha ** 2) * precision * recall / (betha ** 2 * precision + recall)

