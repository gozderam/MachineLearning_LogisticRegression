import pandas as pd
import numpy as np
from measures.evaluation_metrics import get_accuracy, get_f_score, get_recall, get_precision, get_recall


def compare_models(models: list, cmp_models: list, X_train, X_test, y_train, y_test, i=1):
    results = []
    for model_name, model in models:
        acc, f1, p, r = get_model_results(model, X_test, y_test, i)
        results.append([model_name, acc, f1, p, r])

    for cmp_model_name, cmp_model in cmp_models:
        acc, f1, p, r = get_cmp_model_results(cmp_model, X_train, X_test, y_train, y_test, i)
        results.append([cmp_model_name, acc, f1, p, r])

    res = pd.DataFrame(results)
    res.columns = ['model', 'accuracy', 'f1_score', 'precision', 'recall']
    return res.sort_values(by='accuracy', ascending=False)


def get_avg_measures(predicts: list, target):
    acc = 0.0
    f1 = 0.0
    recall = 0.0
    precision = 0.0
    n = len(predicts)
    for i in range(0, n):
        acc += get_accuracy(predicts[i], target)
        f1 += get_f_score(predicts[i], target)
        recall += get_recall(predicts[i], target)
        precision += get_precision(predicts[i], target)

    return acc / n, f1 / n, recall / n, precision / n


def get_model_results(model, X_test, y_test, trials=1):
    predict = []
    for i in range(0, trials):
        predict.append(model.predict(X_test))

    return get_avg_measures(predict, y_test)


def get_cmp_model_results(model, X_train, X_test, y_train, y_test, trials=1):
    predict = []
    for i in range(0, trials):
        model.fit(X_train, y_train)
        predict.append(model.predict(X_test))

    return get_avg_measures(predict, y_test)
    


