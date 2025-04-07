
from asyncio import FastChildWatcher
from cmath import exp
import imp
import json
import time
import os
from pathlib import Path
from token import CIRCUMFLEX
from typing import Dict, List, Optional, Tuple
from urllib.request import UnknownHandler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import data
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import time
import datetime
from evaluate.evaluator import evaluator_new_nogroup
import functools
from models.expomf import ExpoMF
from models.recommenders import  PointwiseRecommender_New,PointwiseRecommender_LGN,PointwiseRecommender_XSimGCL
from models.dibmf import DIBMF

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)


def corr_(x,y):
    #print(x.shape)
    #print(y.shape)
    a = np.array([x,y])
    b = np.corrcoef(a)
    return np.around(b[0,1],3)

def cos_sim(a,b):
    dot_p = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.round(dot_p/(norm_a*norm_b),3)
def pointwise_trainer_new(sess: tf.Session, model: PointwiseRecommender_New,
                        train:np.ndarray,
                      item_freq: np.ndarray,user_freq: np.ndarray,group_num:int,group_list,target_u,target_u_pop,candidate_u,
                      max_iters: int = 1000, batch_size: int = 2**11,
                      model_name: str = 'rmf',dataset_name: str = 'yahooR3',pos_u:np.ndarray =None,train_batch=None,pos_cnt=None,user_batch=None) -> Tuple[np.ndarray, np.ndarray]:
    
    singular_value = [] 
    var_loss = []
    matrix = []
    sv_list = []
    np.set_printoptions(suppress=True)
    metrics = ['N@20','R@20']#,'P@20'
    at_k = [20]
    #metrics = ['NDCG@20']
    #best_result = pd.DataFrame(index = metrics)
    #best_result[model_name] = [0.,0.,0.]
    best_result = []
    best_result_unbias = []
    best_iter = 0
    max_interval  = 100
    max_sum = 0.
    max_sum_unbias = 0.
    best_ib,best_iemb,best_uemb= None,None,None
    max_sum_ =[0,0,0,0,0] 
    best_ans =[0,0,0,0,0]
    best_iter_ = [0,0,0,0,0]
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    num_users = user_freq.shape[0]
    num_items = item_freq.shape[0]
    #best_iemb ,best_uemb = 
    path = Path(f'../data/{dataset_name}/point')
    path.mkdir(parents=True, exist_ok=True)

   
    pos_train = train

    np.random.seed(123)

    item_pop = item_freq #(item_freq/item_freq.sum())
    user_pop = user_freq #(user_freq/user_freq.sum())
    item_pop = item_pop / ((item_pop**2).sum())**0.5
    #user_pop = user_pop / ((user_pop**2).sum())**0.5
    item_pop = np.expand_dims(item_pop,0)
    user_pop = np.expand_dims(user_pop,1)

    #print(item_pop.shape,user_pop[user_batch[0]].shape)
    
    
    
    neg_num = 4 
    n_batch = num_users//batch_size

    #n_batch = 1
    #train_batch = np.zeros((batch_size*(neg_num+1),3))
    print('item:',num_items)
    print('user:',num_users)
    #print('all pos:',num_pos)

    user_matrix = np.zeros((num_users,neg_num))

    for u in range(num_users):
        for i in range(neg_num):
            user_matrix[u][i] = u
    
    user_all = np.arange(num_users).astype(np.int32)
    np.random.shuffle(user_all)

    #user_batch_test  =np.load(f'../data/{dataset_name}/point/'+str(group_num)+'/user_batch_test.npy',allow_pickle=True)
    
    eval_mask  =np.load(f'../data/{dataset_name}/point/'+str(group_num)+'/eval_mask.npy',allow_pickle=True).tolist()
    print(len(eval_mask))
    
    
        
    
    train_batch_neg_list = np.load(f'../data/{dataset_name}/point/'+str(group_num)+'/train_batch_neg_list.npy',allow_pickle=True)
    
    data_len  = len(train_batch_neg_list)
    '''
    if max_iters>data_len:
        for i in range(data_len,max_iters):
            train_batch_neg_list.append([[]for j in range(n_batch+1)])
    '''
    print(data_len,len(train_batch_neg_list))
    #np.save(str(path/str(group_num)/'train_batch_neg_list.npy'),arr=np.array(train_batch_neg_list))

    #print(eval_mask)
    for u in range(1):
        print(np.sum(eval_mask[u]))
    
    #print(train_.shape)
    print('=====start training=====')
    #data_len = 0
 
    
    #print(train_batch_neg_list[0][0][:20,:])
    print(n_batch)
    
    start_time = time.time()
    for i in np.arange(max_iters):
        
        for j in range(n_batch+1):

            _,loss,regterm= sess.run([model.apply_grads,model.loss,model.regloss],
                        feed_dict={model.pos_users: train_batch[j][:, 0],
                                    model.pos_items: train_batch[j][:, 1],
                                    model.neg_items: train_batch_neg_list[i%data_len][j][:,1],
                                    model.neg_users: train_batch_neg_list[i%data_len][j][:,0],
                                    model.uniusers:user_batch[j],
                                    model.item_pop:item_pop, 
                                    model.user_pop:user_pop[user_batch[j]],
                                    model.e_i:np.ones((1,num_items)),
                                    model.e_u:np.ones((1,user_batch[j].shape[0]))})
        

        if i%10 ==0 : 
            print(i,loss,regterm)
            cur_t = time.time()
            print((cur_t-start_time)/60)
            top_item_list = []
            for j in range(n_batch+1):
                top_idx = sess.run([model.top_idx],feed_dict={model.uniusers:user_batch[j],model.mask_matrix:eval_mask[j],model.item_pop:item_pop,model.user_pop:user_pop[user_batch[j]]})
 
                top_item_list.append(top_idx[0])
            
            ret,p = evaluator_new_nogroup(top_item_list=top_item_list,
                                model_name=model_name,  target_u= target_u,target_u_pop= target_u_pop,user_batch=user_batch,group_item = group_list ,at_k=at_k)
            

            if ret.loc[metrics[0],model_name] > max_sum:
                #u_emb,i_emb = sess.run([model.usera_embeddings,model.item_embeddings])
                best_iter = i
                #best_uemb = u_emb
                #best_iemb = i_emb
                #best_ib = i_bias
                max_sum = ret.loc[metrics[0],model_name]
                best_result = [ret,p]
              
            if i - best_iter > max_interval:
                break



    print('final result',best_iter)
    print(best_result)
    
    

    sess.close()
    end_time = time.time()
    print((end_time-start_time)/3600)
    return best_uemb, best_iemb


def sigmoid(x):
    return 1./(1+np.exp(-x))



class Trainer:
    """Trainer Class for ImplicitRecommender."""

    #suffixes = ['all', 'rare']
    at_k = [20]
    #rare_item_threshold = 500

    def __init__(self, max_iters: int = 2000, lam=1e-4, reglam =1e-4,reglam2 =1e-4,batch_size: int = 10,
                 eta: float = 0.01, model_name: str = 'mf',dataset_name: str = 'ml-1m',dim: int = 64,group_num:int=5,alpha0=None,alpha1 :float=0.,reglam_list=None,backbone=None) -> None:
        """Initialize class."""
        
        self.dim = dim
        self.lam = lam
        self.reglam = reglam
        self.reglam2 = reglam2
        self.alpha0 =alpha0
        self.alpha1 = alpha1
        self.group_num = group_num
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.reglam_list = reglam_list
        self.backbone = backbone
    
    def run(self) -> None:
        """Train implicit recommenders."""

        print('======Loading Data======')

        pos_u = None
        group_item = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/group_item.npy',allow_pickle=True).tolist()
        train = np.load(f'../data/{self.dataset_name}/point/train.npy',allow_pickle=True) 
        item_freq = np.load(f'../data/{self.dataset_name}/point/item_freq.npy',allow_pickle=True)
        user_freq = np.load(f'../data/{self.dataset_name}/point/user_freq.npy',allow_pickle=True)
        target_u = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/target_u.npy',allow_pickle=True).tolist()
                
        target_u_pop = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/target_u_pop.npy',allow_pickle=True).tolist()
        
        candidate_u = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/candidate_u.npy',allow_pickle=True).tolist()
        
        train_batch = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/train_batch.npy',allow_pickle=True)
        user_batch = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/user_batch.npy',allow_pickle=True)
        pos_cnt = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/pos_cnt.npy',allow_pickle=True)

        norm_adj = sparse.load_npz(f'../data/{self.dataset_name}/s_norm_adj_mat.npz')
        #train_batch_neg_list = np.load(f'../data/{self.dataset_name}/point/'+str(self.group_num)+'/train_batch_neg_list.npy',allow_pickle=True).tolist()

        
        num_users = user_freq.shape[0]
        num_items = item_freq.shape[0]

        for term in group_item:
           print(len(term))

        for i,cur_reglam in enumerate(self.reglam_list): 
            for j,cur_alpha0 in enumerate(self.alpha0):
                print(cur_alpha0,cur_reglam)
                tf.set_random_seed(12345)
                ops.reset_default_graph()
                #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
                config = tf.ConfigProto(log_device_placement  = False, allow_soft_placement = True)
                config.gpu_options.allow_growth=True
                config.gpu_options.per_process_gpu_memory_fraction = 0.8
                sess = tf.Session(config=config)
                if self.backbone =='mf':
                    model = PointwiseRecommender_New(
                        num_users=num_users, num_items=num_items, dim=self.dim, lam=self.lam, eta=self.eta,reglam=cur_reglam,alpha0 = cur_alpha0,alpha1 =self.alpha1,reglam2=self.reglam2 )
                elif self.backbone =='xsimgcl':
                    model = PointwiseRecommender_XSimGCL(
                        num_users=num_users, num_items=num_items, dim=self.dim, lam=self.lam, eta=self.eta,reglam=cur_reglam,alpha0 = cur_alpha0,alpha1 =self.alpha1,reglam2=self.reglam2,norm_adj =norm_adj )
                else :
                    model = PointwiseRecommender_LGN(
                        num_users=num_users, num_items=num_items, dim=self.dim, lam=self.lam, eta=self.eta,reglam=cur_reglam,alpha0 = cur_alpha0,alpha1 =self.alpha1,reglam2=self.reglam2,norm_adj =norm_adj )
                u_emb, i_emb = pointwise_trainer_new(
                    sess, model=model,  train = train,
                    item_freq = item_freq,  user_freq =user_freq,group_num = self.group_num,group_list = group_item,target_u = target_u,target_u_pop=target_u_pop,candidate_u=candidate_u,
                    max_iters=self.max_iters, batch_size=2**self.batch_size,
                    model_name=self.model_name,dataset_name=self.dataset_name,pos_u =pos_u,train_batch = train_batch,pos_cnt = pos_cnt,user_batch = user_batch)
                print(cur_alpha0,cur_reglam,'done!')
                #flag = True
        
