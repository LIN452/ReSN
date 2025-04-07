"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import yaml
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from data.preprocessor import preprocess_dataset,preprocess_dataset_new
from trainer import Trainer
from visualize import plot_test_curves_with_ranking_metrics

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, choices=['regmf'])
parser.add_argument('--preprocess_data', action='store_true')
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--max_iters',type=int,default=221)
parser.add_argument('--batch_size',type=int,default=11)
parser.add_argument('--lr',type=float,default=0.005)
parser.add_argument('--lam',type=float,default=0.00001)
parser.add_argument('--group_num',type=int,default=5)
parser.add_argument('--cuda',type=str,default='')
parser.add_argument('--reglam_list', nargs='*',type=float, default=[0])

parser.add_argument('--reglam',type=float,default=0.00000)
parser.add_argument('--reglam2',type=float,default=0.00000)
parser.add_argument('--threshold',type=int,default=4)
parser.add_argument('--dim',type=int,default=256)

parser.add_argument('--alpha1',type=float,default=0.)
parser.add_argument('--alpha0', nargs='*',type=float,default=[0])

parser.add_argument('--backbone', type=str, default='mf')

import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(tf.test.is_gpu_available())
    # hyper-parameters
    #config = yaml.safe_load(open('../config.yaml', 'rb'))
    
    threshold = args.threshold
    batch_size = args.batch_size
    max_iters = args.max_iters
    lam = args.lam
    reglam = args.reglam
    reglam2 = args.reglam2
    eta = args.lr
    model_name = args.model_name
    dataset_name = args.dataset
    dim = args.dim
    group_num =args.group_num
    alpha0 = args.alpha0
    alpha1 = args.alpha1
    reglam_list = args.reglam_list
    backbone = args.backbone
    # run simulations
    print(reglam_list)
    if reglam_list ==  None: 
        reglam_list.append(reglam)
   
    trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
                      lam=lam,reglam= reglam,reglam2 = reglam2, eta=eta, model_name=model_name,dataset_name = dataset_name,dim = dim,group_num=group_num,alpha0 = alpha0,alpha1 =alpha1,reglam_list= reglam_list,backbone = backbone)
    trainer.run()

    print('===========Done!=================')
    print(args.dataset)
    print('lam',args.lam)
    print('lr',args.lr) 
    print('reglam',args.reglam)
    print('reglam2',args.reglam2)
    print('alpha0',args.alpha0)
    print('alpha1',args.alpha1)
    print('\n', '=' * 25, '\n')
    print('Finished Running {model_name}!')
    
    
    
    
