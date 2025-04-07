"""Evaluation metrics for Collaborative Filltering with Implicit Feedback."""
from typing import Callable, Dict, List, Optional

import numpy as np

eps = 10e-3  # pscore clipping


def precision_at_k(r, k,maxlen):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    try:
        r = r[:k]
    except:
        print(r)
        raise ImportError('error r')
    return np.mean(r)


def dcg_at_k(r, k,maxlen, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, maxlen, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    tp = 1. / np.log2(np.arange(2, max(k,maxlen) + 2))
    dcg_max = (tp[:min(maxlen, k)]).sum()
    #dcg_max = (tp[:maxlen]).sum()
    
    if not dcg_max:
        return 0.
    r_k = r[:k]
    dcg_at_k_ = (r_k * tp[:k]).sum() 
    return dcg_at_k_ / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = r[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = r[:k]
    return min(1.,np.sum(r))
    

def get_r(user_pos_test, r):
    r_new = np.isin(r,user_pos_test).astype(np.float)
    return r_new

def get_performance(user_pos_test, r, K):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    r = get_r(user_pos_test,r)
    
    precision.append(precision_at_k(r, K))#P = TP/ (TP+FP)
    recall.append(recall_at_k(r, K, len(user_pos_test)))#R = TP/ (TP+FN)
    ndcg.append(ndcg_at_k(r, K, len(user_pos_test)))
    hit_ratio.append(hit_at_k(r, K))#HR = SIGMA(TP) / SIGMA(test_set)
    # print(hit_ratio)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


'''
def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray,
             k: int, pscore: Optional[np.array] = None) -> float:
    """Calculate a DCG score for a given user."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]  #是按照预测的score从大到小排序的真实score！！！

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)  #pscore默认是1

    dcg_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0] / pscore_sorted_by_score[0]
        k = min(k,y_true.shape[0])
        for i in np.arange(1, k):
            dcg_score += y_true_sorted_by_score[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))

    final_score = dcg_score if pscore is None \
        else dcg_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray,
             k: int, pscore: Optional[np.array] = None) -> float:
    """Calculate a DCG score for a given user."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]  #是按照预测的score从大到小排序的真实score！！！
    y_true_sorted_by_score2 = y_true[y_true.argsort()[::-1]]  

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)  #pscore默认是1

    dcg_score = 0.0
    idcg_score = 0.0
    
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0] / pscore_sorted_by_score[0]
        
        idcg_score += y_true_sorted_by_score2[0] / pscore_sorted_by_score[0]
        k = min(k,y_true.shape[0])
        for i in np.arange(1, k):
            dcg_score += y_true_sorted_by_score[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))
            
            idcg_score += y_true_sorted_by_score2[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))
            
                

    final_score = dcg_score if pscore is None \
        else dcg_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score / idcg_score



def average_precision_at_k(y_true: np.ndarray, y_score: np.ndarray,
                           k: int, pscore: Optional[np.array] = None) -> float:
    """Calculate a average precision for a given user."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    average_precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        for i in np.arange(k):
            if y_true_sorted_by_score[i] == 1:
                average_precision_score += \
                    np.sum(y_true_sorted_by_score[:i + 1] /
                           pscore_sorted_by_score[:i + 1]) / (i + 1)

    final_score = average_precision_score if pscore is None \
        else average_precision_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray,
                k: int, pscore: Optional[np.array] = None) -> float:
    """Calculate a recall score for a given user."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    recall_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        recall_score = np.sum(
            y_true_sorted_by_score[:k] / pscore_sorted_by_score[:k])   

    final_score = recall_score / np.sum(y_true) if pscore is None \
        else recall_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray,
                k: int, pscore: Optional[np.array] = None) -> float:
    """Calculate a recall score for a given user."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        precision_score = np.sum(
            y_true_sorted_by_score[:k] / pscore_sorted_by_score[:k])   

    final_score = precision_score / k if pscore is None \
        else precision_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score
'''