
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from io import SEEK_CUR

import numpy as np
from numpy.core.fromnumeric import clip
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()



class PointwiseRecommender_New(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float,reglam:float,alpha0:float,alpha1:float,reglam2:float) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.reglam = reglam
        self.reglam2 = reglam2
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.eps =1e-5
        self.neg_var = 200
        self.neg_num = 4
        self.top_k = 20
        #self.wd = 0.99
        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.pos_users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.uniusers = tf.placeholder(tf.int32, [None], name='uniuser_placeholder')
        self.uniitems = tf.placeholder(tf.int32, [None], name='uniitem_placeholder')
        #self.test_users = tf.placeholder(tf.int32, [None], name='testuser_placeholder')
        #self.test_items = tf.placeholder(tf.int32, [None], name='testitem_placeholder')
        #self.wd = tf.placeholder(tf.float32,[None],name = 'weight_dacay')
        self.neg_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'neg_matrix')
        self.user_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'user_matrix')
        self.mask_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'mask_matrix')
        self.label_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'label_matrix')
        #self.top_k = tf.placeholder(tf.int32,[None],name = 'top_k')
        self.neg_items = tf.placeholder(tf.int32,[None],name = 'neg_item')
        self.neg_users = tf.placeholder(tf.int32,[None],name = 'neg_user')
        self.e_i = tf.placeholder(tf.float32, [1,None], name='e_i')
        self.e_u = tf.placeholder(tf.float32, [1,None], name='e_u')
        self.item_pop = tf.placeholder(
            tf.float32, [1,None], name='item_pop')
        
        self.user_pop = tf.placeholder(
            tf.float32, [None, 1], name='user_pop')
        self.randr = tf.placeholder(
            tf.float32, [None, 1], name='randr')
        
        #self.labels = tf.placeholder(
        #    tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            '''
            self.user_bias_rel = tf.get_variable(
                'user_bias_rel', shape=[self.num_users, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_bias_rel = tf.get_variable(
                'item_bias_rel', shape=[self.num_items, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            '''
            # lookup embeddings of current batch
            self.pos_u_embed = tf.nn.embedding_lookup(
                self.user_embeddings, self.pos_users)
            self.pos_i_embed = tf.nn.embedding_lookup(
                self.item_embeddings, self.pos_items)
            
            self.var_neg_emb  = tf.nn.embedding_lookup(
                self.item_embeddings, self.neg_matrix)
            self.var_u_emb = tf.nn.embedding_lookup(
                self.user_embeddings, self.user_matrix)
            
            self.neg_i_emb  = tf.nn.embedding_lookup(
                self.item_embeddings, self.neg_items)
            self.neg_u_emb = tf.nn.embedding_lookup(
                self.user_embeddings, self.neg_users)
            
            self.u_embed =  tf.nn.embedding_lookup(
                self.user_embeddings, self.uniusers)
            
  
            
            self.all_matrix = tf.matmul(self.u_embed,tf.transpose(self.item_embeddings))
            
            self.matrix_ = tf.sigmoid(tf.matmul(self.u_embed,tf.transpose(self.item_embeddings)))
            

            self.top_idx = tf.nn.top_k(self.matrix_*self.mask_matrix,k=self.top_k).indices
 
            self.UVe = tf.expand_dims(tf.reduce_sum(self.matrix_,axis = 0),1)
            
            
            self.q = tf.stop_gradient(self.UVe/tf.sqrt(tf.reduce_sum(tf.square(self.UVe))))
            
            
            '''
            self.u_bias_rel = tf.nn.embedding_lookup( 
                self.user_bias_rel, self.users)
            self.i_bias_rel = tf.nn.embedding_lookup(
                self.item_bias_rel, self.items)
            '''
        with tf.variable_scope('prediction'):
            
            # self.pos_u_embed_norm = tf.nn.l2_normalize(self.pos_u_embed,axis = 0)
            # self.pos_i_embed_norm = tf.nn.l2_normalize(self.pos_i_embed,axis = 0)
            # self.neg_u_embed_norm = tf.nn.l2_normalize(self.neg_u_emb,axis = 0)
            # self.neg_i_embed_norm = tf.nn.l2_normalize(self.neg_i_emb,axis = 0)
            
            #self.pos_preds_norm = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pos_u_embed_norm, self.pos_i_embed), 1),1)  #[train_batch,1]
            
            #self.neg_preds_norm =tf.expand_dims(tf.reduce_sum(tf.multiply(self.neg_u_embed_norm,self.neg_i_embed_norm),1),1)  #[train_batch*neg_num,1]
            
            self.pos_preds = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pos_u_embed, self.pos_i_embed), 1),1)  #[train_batch,1]
            
            self.neg_preds =tf.expand_dims(tf.reduce_sum(tf.multiply(self.neg_u_emb,self.neg_i_emb),1),1)  #[train_batch*neg_num,1]
            #self.var_preds =tf.reduce_sum(tf.multiply(self.var_u_emb,self.var_neg_emb),2)
            
            self.pos_preds_sig = tf.sigmoid(self.pos_preds)
            self.neg_preds_sig = tf.sigmoid(self.neg_preds)
            '''
            self.preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            
            self.sig_preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            '''
            #self.scores = tf.matmul(self.u_embed_test,tf.transpose(self.i_embed_test.T))

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            
           

 
            self.regterm = tf.matmul(self.matrix_,self.q)

            self.regloss = tf.reduce_sum(tf.square(self.regterm))

            

            self.weighted_mse = tf.reduce_sum(tf.square(1-self.pos_preds_sig))+ tf.reduce_sum(tf.square(self.neg_preds_sig))
            
            self.weighted_bce = -tf.reduce_sum(tf.log(self.pos_preds_sig+self.eps))-tf.reduce_sum(tf.log(1-self.neg_preds_sig+self.eps))
            
            self.weighted_bpr = -tf.reduce_sum(tf.log(tf.sigmoid(self.pos_preds-self.neg_preds)))

            
            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds+self.reglam*(self.regloss) #+self.alpha0*(self.var_term_new_neg+self.reglam2*self.var_term_new_pos)
            #+ self.alpha0*self.var_term0+self.alpha1*self.var_term1 +self.reglam2*self.regloss4

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)


class PointwiseRecommender_LGN(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float,reglam:float,alpha0:float,alpha1:float,reglam2:float,norm_adj) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.reglam = reglam
        self.reglam2 = reglam2
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.eps =1e-6
        self.neg_var = 200
        self.neg_num = 4
        self.top_k = 20
        self.norm_adj  = norm_adj
        self.n_layers = 3
        self.n_fold = 10
        self.weight_size = eval('[10, 10, 10]')
        self.node_dropout = eval('0.1') # args.node_dropout[0]
        self.mess_dropout = eval('[0.1,0.1,0.1]') # args.mess_dropout
        self.batch_size =2048
        
        #self.wd = 0.99
        # Build the graphs
        self.create_placeholders()
        self.weights = self._init_weights()
        self.usera_embeddings, self.itema_embeddings = self._create_lightgcn_embed()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()
    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.dim]), name='item_embedding')
                           
        self.weight_size_list = [self.dim] + self.weight_size
        
        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        #self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        #A_hat = self.sparse_norm_adj
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):
            
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            
            all_embeddings += [ego_embeddings]
            
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings

    
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.pos_users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.uniusers = tf.placeholder(tf.int32, [None], name='uniuser_placeholder')
        self.uniitems = tf.placeholder(tf.int32, [None], name='uniitem_placeholder')
        #self.test_users = tf.placeholder(tf.int32, [None], name='testuser_placeholder')
        #self.test_items = tf.placeholder(tf.int32, [None], name='testitem_placeholder')
        #self.wd = tf.placeholder(tf.float32,[None],name = 'weight_dacay')
        self.neg_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'neg_matrix')
        self.user_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'user_matrix')
        self.mask_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'mask_matrix')
        self.label_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'label_matrix')
        #self.top_k = tf.placeholder(tf.int32,[None],name = 'top_k')
        self.neg_items = tf.placeholder(tf.int32,[None],name = 'neg_item')
        self.neg_users = tf.placeholder(tf.int32,[None],name = 'neg_user')
        self.e_i = tf.placeholder(tf.float32, [1,None], name='e_i')
        self.e_u = tf.placeholder(tf.float32, [1,None], name='e_u')
        self.item_pop = tf.placeholder(
            tf.float32, [1,None], name='item_pop')
        
        self.user_pop = tf.placeholder(
            tf.float32, [None, 1], name='user_pop')
        self.randr = tf.placeholder(
            tf.float32, [None, 1], name='randr')
        
        #self.labels = tf.placeholder(
        #    tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            # self.user_embeddings = tf.get_variable(
            #     'user_embeddings', shape=[self.num_users, self.dim],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.item_embeddings = tf.get_variable(
            #     'item_embeddings', shape=[self.num_items, self.dim],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # '''
            # self.user_bias_rel = tf.get_variable(
            #     'user_bias_rel', shape=[self.num_users, 1],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.item_bias_rel = tf.get_variable(
            #     'item_bias_rel', shape=[self.num_items, 1],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # '''
            # lookup embeddings of current batch
            self.pos_u_embed = tf.nn.embedding_lookup(
                self.usera_embeddings, self.pos_users)
            self.pos_i_embed = tf.nn.embedding_lookup(
                self.itema_embeddings, self.pos_items)
            
            # self.var_neg_emb  = tf.nn.embedding_lookup(
            #     self.item_embeddings, self.neg_matrix)
            # self.var_u_emb = tf.nn.embedding_lookup(
            #     self.user_embeddings, self.user_matrix)
            
            self.neg_i_emb  = tf.nn.embedding_lookup(
                self.itema_embeddings, self.neg_items)
            self.neg_u_emb = tf.nn.embedding_lookup(
                self.usera_embeddings, self.neg_users)
            
            self.u_embed =  tf.nn.embedding_lookup(
                self.usera_embeddings, self.uniusers)
            
            self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.uniusers)
            self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
            self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)


            self.all_matrix = tf.matmul(self.u_embed,tf.transpose(self.item_embeddings))
            
            self.matrix_ = tf.sigmoid(tf.matmul(self.u_embed,tf.transpose(self.item_embeddings)))
            

            self.top_idx = tf.nn.top_k(self.matrix_*self.mask_matrix,k=self.top_k).indices
 
            self.UVe = tf.expand_dims(tf.reduce_sum(self.matrix_,axis = 0),1)
            
            
            self.q = tf.stop_gradient(self.UVe/tf.sqrt(tf.reduce_sum(tf.square(self.UVe))))
            
        with tf.variable_scope('prediction'):
            self.pos_preds = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pos_u_embed, self.pos_i_embed), 1),1)  #[train_batch,1]
            
            self.neg_preds =tf.expand_dims(tf.reduce_sum(tf.multiply(self.neg_u_emb,self.neg_i_emb),1),1)  #[train_batch*neg_num,1]
            #self.var_preds =tf.reduce_sum(tf.multiply(self.var_u_emb,self.var_neg_emb),2)
            
            self.pos_preds_sig = tf.sigmoid(self.pos_preds)
            self.neg_preds_sig = tf.sigmoid(self.neg_preds)
            '''
            self.preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            
            self.sig_preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            '''
            #self.scores = tf.matmul(self.u_embed_test,tf.transpose(self.i_embed_test.T))

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            
            self.regterm = tf.matmul(self.matrix_,self.q)

            self.regloss = tf.reduce_sum(tf.square(self.regterm))

            

            self.weighted_mse = tf.reduce_sum(tf.square(1-self.pos_preds_sig))+ tf.reduce_sum(tf.square(self.neg_preds_sig))
            
            self.weighted_bce = -tf.reduce_sum(tf.log(self.pos_preds_sig+self.eps))-tf.reduce_sum(tf.log(1-self.neg_preds_sig+self.eps))
            
            self.weighted_bpr = -tf.reduce_sum(tf.log(tf.sigmoid(self.pos_preds-self.neg_preds)))
            
            
            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)+ tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds+self.reglam*(self.regloss) #+self.alpha0*(self.var_term_new_neg+self.reglam2*self.var_term_new_pos)
            #+ self.alpha0*self.var_term0+self.alpha1*self.var_term1 +self.reglam2*self.regloss4

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)




class PointwiseRecommender_XSimGCL(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float,reglam:float,alpha0:float,alpha1:float,reglam2:float,norm_adj) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.reglam = reglam
        self.reglam2 = reglam2
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.eps =1e-6
        self.neg_var = 200
        self.neg_num = 4
        self.top_k = 20
        self.norm_adj  = norm_adj
        self.n_layers = 2
        self.n_fold = 200
        self.weight_size = eval('[10, 10, 10]')
        self.node_dropout = eval('0.1') # args.node_dropout[0]
        self.mess_dropout = eval('[0.1,0.1,0.1]') # args.mess_dropout
        self.batch_size =2048
        self.eps_gcl =0.1
        self.layer_cl = 2
        #self.wd = 0.99
        # Build the graphs
        self.create_placeholders()
        self.weights = self._init_weights()
        self.A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        
        self.usera_embeddings, self.itema_embeddings,self.userp1_embeddings,self.itemp1_embeddings= self._create_embed(True)
        self.user_rate_embeddings ,self.item_rate_embeddings=self._create_embed(False)
        
        
        #self.userp1_embeddings, self.itemp1_embeddings = self._create_perturbed_embed()
        #self.userp2_embeddings, self.itemp2_embeddings = self._create_perturbed_embed()
        self.item_embedding_rating  = self.itema_embeddings
        self.build_graph()
        self.create_losses()
        self.add_optimizer()
        
    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.dim]), name='item_embedding')
                           
        self.weight_size_list = [self.dim] + self.weight_size
        
        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        #self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
       # A_hat_dense = tf.sparse_tensor_to_dense(A_hat,validate_indices=False)
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):
            
            # temp_embed = []
            # for f in range(self.n_fold):
            #     temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # side_embeddings = tf.concat(temp_embed, 0)
            # ego_embeddings = side_embeddings
            
            ego_embeddings = tf.sparse_tensor_dense_matmul(A_hat, ego_embeddings)
            #ego_embeddings = tf.matmul(A_hat_dense, ego_embeddings,a_is_sparse=True)
            
            all_embeddings += [ego_embeddings]
            
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_embed(self,perturbed=False):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        #self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        #A_hat_dense = tf.sparse_tensor_to_dense(A_hat,validate_indices=False)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):
            
            # temp_embed = []
            # for f in range(self.n_fold):
            #     temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # side_embeddings = tf.concat(temp_embed, 0)
            # ego_embeddings = side_embeddings
            
            ego_embeddings = tf.sparse_tensor_dense_matmul(A_hat, ego_embeddings)
            #ego_embeddings = tf.matmul(A_hat_dense, ego_embeddings,a_is_sparse=True)
            if perturbed:
                random_noise = tf.random_uniform(ego_embeddings.shape)
                ego_embeddings += tf.multiply(tf.sign(ego_embeddings),tf.nn.l2_normalize(random_noise, 1)) * self.eps_gcl
            all_embeddings += [ego_embeddings]
            if k == self.layer_cl-1 :
                all_embeddings_cl =ego_embeddings
            
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        u_g_embeddings_cl, i_g_embeddings_cl = tf.split(all_embeddings_cl, [self.num_users, self.num_items], 0)
        if perturbed:
            return u_g_embeddings, i_g_embeddings,u_g_embeddings_cl,i_g_embeddings_cl
        
        return u_g_embeddings, i_g_embeddings

    
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.pos_users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.uniusers = tf.placeholder(tf.int32, [None], name='uniuser_placeholder')
        self.uniitems = tf.placeholder(tf.int32, [None], name='uniitem_placeholder')
        
        self.neg_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'neg_matrix')
        self.user_matrix = tf.placeholder(tf.int32,[None,self.neg_var],name = 'user_matrix')
        self.mask_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'mask_matrix')
        self.label_matrix = tf.placeholder(tf.float32,[None,self.num_items],name = 'label_matrix')
        #self.top_k = tf.placeholder(tf.int32,[None],name = 'top_k')
        self.neg_items = tf.placeholder(tf.int32,[None],name = 'neg_item')
        self.neg_users = tf.placeholder(tf.int32,[None],name = 'neg_user')
        self.e_i = tf.placeholder(tf.float32, [1,None], name='e_i')
        self.e_u = tf.placeholder(tf.float32, [1,None], name='e_u')
        self.item_pop = tf.placeholder(tf.float32, [1,None], name='item_pop')
        
        self.user_pop = tf.placeholder(tf.float32, [None, 1], name='user_pop')
        self.randr = tf.placeholder(tf.float32, [None, 1], name='randr')
        
        #self.labels = tf.placeholder(
        #    tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            # self.user_embeddings = tf.get_variable(
            #     'user_embeddings', shape=[self.num_users, self.dim],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.item_embeddings = tf.get_variable(
            #     'item_embeddings', shape=[self.num_items, self.dim],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # '''
            # self.user_bias_rel = tf.get_variable(
            #     'user_bias_rel', shape=[self.num_users, 1],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.item_bias_rel = tf.get_variable(
            #     'item_bias_rel', shape=[self.num_items, 1],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # '''
            # lookup embeddings of current batch
            self.pos_u_embed = tf.nn.embedding_lookup(self.usera_embeddings, self.pos_users)
            self.pos_i_embed = tf.nn.embedding_lookup(self.itema_embeddings, self.pos_items)
            
            self.neg_i_emb  = tf.nn.embedding_lookup(self.itema_embeddings, self.neg_items)
            self.neg_u_emb = tf.nn.embedding_lookup(self.usera_embeddings, self.neg_users)
            
            self.u_embed =  tf.nn.embedding_lookup(self.user_rate_embeddings, self.uniusers)
            
            self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.uniusers)
            self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
            self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

            self.all_matrix = tf.matmul(self.u_embed,tf.transpose(self.item_embeddings))
            
            self.matrix_ = tf.sigmoid(tf.matmul(self.u_embed,tf.transpose(self.item_embeddings)))
            

            self.top_idx = tf.nn.top_k(self.matrix_*self.mask_matrix,k=self.top_k).indices
 
            self.UVe = tf.expand_dims(tf.reduce_sum(self.matrix_,axis = 0),1)
            
            
            self.q = tf.stop_gradient(self.UVe/tf.sqrt(tf.reduce_sum(tf.square(self.UVe))))
                        
            
        with tf.variable_scope('prediction'):
            self.pos_preds = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pos_u_embed, self.pos_i_embed), 1),1)  #[train_batch,1]
            
            self.neg_preds =tf.expand_dims(tf.reduce_sum(tf.multiply(self.neg_u_emb,self.neg_i_emb),1),1)  #[train_batch*neg_num,1]
            #self.var_preds =tf.reduce_sum(tf.multiply(self.var_u_emb,self.var_neg_emb),2)
            
            self.pos_preds_sig = tf.sigmoid(self.pos_preds)
            self.neg_preds_sig = tf.sigmoid(self.neg_preds)
            '''
            self.preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            
            self.sig_preds = tf.sigmoid(
                self.preds, name='sigmoid_prediction')
            '''
            #self.scores = tf.matmul(self.u_embed_test,tf.transpose(self.i_embed_test.T))
    def calc_cl_loss(self):
        
        self.u_emdp1 = tf.nn.embedding_lookup(self.userp1_embeddings, tf.unique(self.pos_users)[0])
        self.u_emdp2 = tf.nn.embedding_lookup(self.usera_embeddings, tf.unique(self.pos_users)[0])
        self.i_emdp1 = tf.nn.embedding_lookup(self.itemp1_embeddings, tf.unique(self.uniitems)[0])
        self.i_emdp2 = tf.nn.embedding_lookup(self.itema_embeddings, tf.unique(self.uniitems)[0])
        norm_u1 = tf.nn.l2_normalize(self.u_emdp1,1)
        norm_u2 = tf.nn.l2_normalize(self.u_emdp2,1)
        norm_i1 = tf.nn.l2_normalize(self.i_emdp1,1)
        norm_i2 = tf.nn.l2_normalize(self.i_emdp2,1)
        
        pos_score_u = tf.reduce_sum(tf.multiply(norm_u1, norm_u2), axis=1)#zu*zu
        pos_score_i = tf.reduce_sum(tf.multiply(norm_i1, norm_i2), axis=1)#zi*zi
        
        ttl_score_u = tf.matmul(norm_u1, norm_u2, transpose_a=False, transpose_b=True)
        ttl_score_i = tf.matmul(norm_i1, norm_i2, transpose_a=False, transpose_b=True)
        
        pos_score_u = tf.exp(pos_score_u / 0.2)
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u / 0.2), axis=1)
        
        pos_score_i = tf.exp(pos_score_i / 0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i / 0.2), axis=1)
        
        cl_loss = -tf.reduce_sum(tf.log(pos_score_u / ttl_score_u)) - tf.reduce_sum(tf.log(pos_score_i / ttl_score_i))

        return cl_loss
        
    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):

            self.regterm = tf.matmul(self.matrix_,self.q)

            self.regloss = tf.reduce_sum(tf.square(self.regterm))

            

            self.weighted_mse = tf.reduce_sum(tf.square(1-self.pos_preds_sig))+ tf.reduce_sum(tf.square(self.neg_preds_sig))
            
            self.weighted_bce = -tf.reduce_sum(tf.log(self.pos_preds_sig+self.eps))-tf.reduce_sum(tf.log(1-self.neg_preds_sig+self.eps))
            
            self.weighted_bpr = -tf.reduce_sum(tf.log(tf.sigmoid(self.pos_preds-self.neg_preds)))
            
            self.cl_loss = self.calc_cl_loss()
           
            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)+ tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds+self.reglam*(self.regloss)+self.alpha0*self.cl_loss #+self.alpha0*(self.var_term_new_neg+self.reglam2*self.var_term_new_pos)
            #+ self.alpha0*self.var_term0+self.alpha1*self.var_term1 +self.reglam2*self.regloss4

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)
