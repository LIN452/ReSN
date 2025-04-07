"""Evaluate Implicit Recommendation models."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.ma.extras import flatten_inplace
import pandas as pd

from .metrics import  dcg_at_k, ndcg_at_k, recall_at_k,precision_at_k,get_performance,get_r

def sigmoid(x):
    return 1./(1+np.exp(-x))

class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray , user_bias:np.ndarray = None, item_bias:np.ndarray = None) -> None:
        """Initialize Class."""
        # latent embeddings
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.user_bias = user_bias
        self.item_bias = item_bias
        

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user

        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1]) # 1*dim
        item_emb = self.item_embed[items]  #n* dim

       
        scores = (user_emb @ item_emb.T).flatten()  # 1*dim * dim* n  ->  1*n
        #
        #scores = sigmoid(scores)
        #print(scores.shape)
        if self.item_bias is not None:
            item_b = self.item_bias[items].flatten()
            scores = (item_b + scores)
        #[xxxx]
        return scores


metrics = {'N':ndcg_at_k,
           'R': recall_at_k#,'P':precision_at_k
           }

def evaluator_new(top_item_list,
                  model_name: str,
                  target_u,
                  target_u_pop,
                  user_batch,
                  at_k: List[int] = [1, 3, 5],
                  user_bias:np.ndarray = None,
                  item_bias:np.ndarray = None) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    
    # define model
    #model = PredictRankings(user_embed=user_embed, item_embed=item_embed,user_bias = user_bias,item_bias = item_bias)

    # prepare ranking metrics
    results = {}
    group_num = len(target_u_pop[0])
    res = []
    
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []
    results['PopQ']= []
    for j in range(group_num):
        result = {}
        for k in at_k:
            for metric in metrics:
                result[f'{metric}@{k}'] = []
        res.append(result)
    # calculate ranking metrics
    #np.random.seed(12345) 
    num_users = len(target_u)
    #cnt = [0 for _  in range(group_num)]
    n_batch = len(top_item_list)
    for e in range(n_batch):
        user_cnt = user_batch[e].shape[0]
        for uu in range(user_cnt):
            user = user_batch[e][uu]
            if len(target_u[user]) ==0:
                continue
            #user_ = np.ones(len(candidate_u[user]))*user
            #item_ = np.array(candidate_u[user])
            #print(user_.shape)
            #scores = sess.run([model.preds],feed_dict={model.users:user_.T,model.items:item_.T})
            #print(scores.shape)
            #scores = model.predict(users=user, items=np.array(candidate_u[user]))
            #top_item = np.argsort(-scores)# from large to little
            #scores = np.array(scores)
            #top_item = np.array(candidate_u[user])[np.argsort(-scores)]# from large to little
            top_item = top_item_list[e][uu]
            r = get_r(target_u[user],top_item)
            
            #print(top_item.shape)
            #print(r.shape)
            if np.sum(r) !=0:
                pos_item = top_item[r == True]
                results['PopQ'].append(target_u[user].index(pos_item[0])/len(target_u[user]))
            
            for k in at_k:
                for metric, metric_func in metrics.items():
                    results[f'{metric}@{k}'].append(metric_func(r,k,len(target_u[user])))
            
            for j in range(group_num):
                if len(target_u_pop[user][j])== 0:
                    continue
                r = get_r(target_u_pop[user][j],top_item)
                #t = get_r(item_list[j],top_item)
                #k = 10
                #cnt[j] += np.sum(t[:k])
                for k in at_k:
                    for metric, metric_func in metrics.items():
                        res[j][f'{metric}@{k}'].append(metric_func(r,k,len(target_u_pop[user][j])))
                
            # aggregate results
    results_df = pd.DataFrame(index=results.keys())
    results_df[model_name] = \
        list(map(np.mean, list(results.values())))
    print(results_df)
    res_df = []
    #cnt_sum = sum(cnt)
    for j in range(group_num):
        result_df = pd.DataFrame(index=res[j].keys())
        result_df[model_name+str(j)] = list(map(np.mean, list(res[j].values())))
        print(result_df)
        res_df.append(result_df)
        #cnt[j]=cnt[j]/cnt_sum
    #results_df[model_name]
    #print(np.around(cnt,5))
    return results_df,res_df
def evaluator_new_nogroup(top_item_list,
                  model_name: str,
                  target_u,
                  target_u_pop,
                  user_batch,group_item,
                  at_k: List[int] = [1, 3, 5],
                  user_bias:np.ndarray = None,
                  item_bias:np.ndarray = None) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    
    # define model
    #model = PredictRankings(user_embed=user_embed, item_embed=item_embed,user_bias = user_bias,item_bias = item_bias)

    # prepare ranking metrics
    results = {}
    #group_num = len(target_u_pop[0])
    res = []
    group_num = len(target_u_pop[0])
    
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []
    
    # calculate ranking metrics
    #np.random.seed(12345) 
    num_users = len(target_u)
    #cnt = [0 for _  in range(group_num)]
    n_batch = len(top_item_list)
    percent = 0.
    tot  = 0
    p_list = [0 for i in range(group_num)]
    
    for e in range(n_batch):
        user_cnt = user_batch[e].shape[0]
        for uu in range(user_cnt):
            user = user_batch[e][uu]
            top_item = top_item_list[e][uu]
            
            tot = tot+1
            #percent += np.sum(get_r(group_item[0],top_item))+np.sum(get_r(group_item[1],top_item))
            
            for gi in range(group_num):
                p_list[gi] +=  np.sum(get_r(group_item[gi],top_item))
            
            if len(target_u[user]) ==0:
                continue

            r = get_r(target_u[user],top_item)
            #print(top_item.shape)
            #print(r.shape)
            '''
            if np.sum(r) !=0:
                pos_item = top_item[r == True]
                results['PopQ'].append(target_u[user].index(pos_item[0])/len(target_u[user]))
            '''
            for k in at_k:
                for metric, metric_func in metrics.items():
                    results[f'{metric}@{k}'].append(metric_func(r,k,len(target_u[user])))
            '''
            for j in range(group_num):
                if len(target_u_pop[user][j])== 0:
                    continue
                r = get_r(target_u_pop[user][j],top_item)
                #t = get_r(item_list[j],top_item)
                #k = 10
                #cnt[j] += np.sum(t[:k])
                for k in at_k:
                    for metric, metric_func in metrics.items():
                        res[j][f'{metric}@{k}'].append(metric_func(r,k,len(target_u_pop[user][j])))
            '''   
            # aggregate results
    #print(num_users,tot)
    results_df = pd.DataFrame(index=results.keys())
    results_df[model_name] = \
        list(map(np.mean, list(results.values())))
    print('P:',np.round((p_list[0]+p_list[1])/tot,3))
    ret = []

    percent = (p_list[0]+p_list[1])/tot
    p_list = [x/tot for x in p_list]
    
    print('P-list:',np.round(p_list,3))
    p_list = p_list/np.sum(p_list)
    print('RR:',np.round(p_list,3),np.round(np.std(p_list),3))
    ret.append(percent)
    ret.append(np.round(p_list,3))
    print(results_df)
    '''
    res_df = []
    #cnt_sum = sum(cnt)
    for j in range(group_num):
        result_df = pd.DataFrame(index=res[j].keys())
        result_df[model_name+str(j)] = list(map(np.mean, list(res[j].values())))
        print(result_df)
        res_df.append(result_df)
        #cnt[j]=cnt[j]/cnt_sum
    #results_df[model_name]
    #print(np.around(cnt,5))
    '''
    return results_df,ret



def unbiased_evaluator(user_embed: np.ndarray,
                       item_embed: np.ndarray,
                       train: np.ndarray,
                       val: np.ndarray,
                       pscore: np.ndarray,
                       metric: str = 'DCG',
                       num_negatives: int = 100,
                       k: int = 5) -> float:
    """Calculate ranking metrics by unbiased evaluator."""
    # validation data
    users = val[:, 0]
    items = val[:, 1]
    positive_pairs = np.r_[train[train[:, 2] == 1, :2], val]

    # define model
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # calculate an unbiased DCG score
    dcg_values = list()
    unique_items = np.unique(items)
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user]
        if all_pos_items == None:
            continue
        neg_items = np.random.permutation(
            np.setdiff1d(unique_items, all_pos_items))[:num_negatives]
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[used_items]
        relevances = np.r_[np.ones_like(pos_items), np.zeros(num_negatives)]

        # calculate an unbiased DCG score for a user
        scores = model.predict(users=user, items=used_items)
        dcg_values.append(metrics[metric](relevances, scores, k, pscore_))

    return np.mean(dcg_values)
