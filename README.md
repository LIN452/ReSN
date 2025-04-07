# ReSN
The implementation of the paper "How Do Recommendation Models Amplify Popularity Bias? An Analysis from the Spectral Perspective" from WSDM 2025.

Paper Link: https://dl.acm.org/doi/10.1145/3701551.3703579

## Instructions on Runing ReSN

### Step1: Data Preparation 
Run src/data/process.ipynb to preprocess the data. 

Run src/data/generate_unbias.ipynb to generate unbiased dataset

Run src/data/sample.ipynb to pre-generate training samples

### Step2: Training the model

Use the following command to train the model:
```
python main.py regmf --lr 0.002 --dataset ml-1m --reglam_list 0.01 --alpha0 0 --dim 256
```
