import math


def ndcg(k, pred, ground_truth) -> float:
    pred_len = min(k, len(pred))
    ground_truth_len = min(k, len(ground_truth))
    denom = [1 / math.log2(i + 2) for i in range(k)]
    dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
    idcg = sum(denom[:ground_truth_len])

    return dcg / idcg


def mape(k, pred, ground_truth) -> float:
    length = min(k, len(pred))
    max_good = min(k, len(ground_truth))
    if len(ground_truth) == 0 or len(pred) == 0:
        return 0
    tp_cum = 0
    result = 0
    for i in range(length):
        if pred[i] in ground_truth:
            tp_cum += 1
            result += tp_cum / ((i + 1) * max_good)
    return result
