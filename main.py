#_*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np

from model import CVRModel

import pdb 

"""
Due to the data anonymity, we utilize the generated fake data for code functionality verification. 
The code assumes that the training phase has been completed and now enters the fine-tuning phase.

Parameter settings: 
- batch_size : the size of training batch 
- user_num : the user number in training batch 
- ad_dims : the embedding size of ad features 
- user_dims : the embedding size of user features 
- learning_rate_small : the learning rate of TransBlock 
- learning_rate_large : the learning rate of original model 
"""
batch_size = 10 
user_num   = 2 
ad_dims    = 100 
user_dims  = 100 
layer_args = [512, 256, 128, 64, 32, 2] 
learning_rate_small = 1e-5 
learning_rate_large = 1e-3 

def fake_data_generator():
    """
    Generating the fake data for finetuning. 

    - ad_embs: [batch_size, ad_dims]
    - user_embs: [user_num, user_dims]
    - indicator: This variable is utilized to indicates the user_embs to the corrosponding ad_embs, [batch_size] 
                 For example, the user_embs is [u0, u1], ad_embs is [a0, a1, a2, a3, a4] and indicator 为 [0, 0, 0, 1, 1]. 
                 It means that the finetuning data pair is that [a0 u0, a1 u0, a2 u0, a3 u1, a4 u1]. 
    - labels: the generated fake labels. 
    """
    # features 
    ad_embs = tf.random.uniform((batch_size, ad_dims), dtype=tf.float32)
    user_embs = tf.random.uniform((user_num, user_dims), dtype=tf.float32)
    indicator = tf.constant(
        np.random.randint(low=0, high=user_num, size=batch_size, dtype=int),
        dtype=tf.int32 
    ) 
    # labels 
    label_seed = np.random.random(size=batch_size) 
    labels = tf.constant(label_seed>0.5, dtype=tf.int32) 

    return user_embs, ad_embs, indicator, labels

def DSC(C, q, lamda, h, dim): 
    '''
    Achieving the Equation 14 in the paper. 

    q: M'_{\hat{y}}, predition scores from f_\Theta 
    C: M'_{\hat{y}|y}, confusion matrix from historical Data 
    lamda: the weight for regularization 
    h: label distribution from Historical Data 
    dim: the size of confusion matrix. It is notable that we could compute importance weight for each instance. 
    ''' 
    I_matrix = np.matrix(np.identity(dim)) 

    # return (C.T * C + lamda * I_matrix).I * (C.T * q + lamda * h) 
    return np.matmul((np.matmul(C.T, C) + lamda * I_matrix).I, (np.matmul(C.T, q) + lamda * h)) 

def train():
    # fake data 
    user_embs_cur, ad_embs_cur, indicator_cur, _          = fake_data_generator() # the first 10-hour data without label of recent day 
    user_embs_h10, ad_embs_h10, indicator_h10, labels_h10 = fake_data_generator() # the first 10-hour data with label of retrieved historical days 
    user_embs_his, ad_embs_his, indicator_his, labels_his = fake_data_generator() # the whole data with label of retrieved historical days 
    
    # model building 
    model = CVRModel(name='cvr_model', input_size=ad_dims+user_dims, layer_args=layer_args)

    ### Step 1: Obtaining the B'_h(\hat{y}|y)与B_h(\hat{y}) 

    # B_h(\hat{y}), we could obtain it from the traffic log in the actual application. 
    y_c = model(user_embs_cur, ad_embs_cur, indicator_cur, 'original', is_training=False) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        y_c_list = [] 
        for _ in range(10): 
            y_c_list.append(sess.run(y_c)) 
    y_c_stat = np.concatenate(y_c_list, 0).mean(0) 
    y_c_stat = np.expand_dims(y_c_stat, 1)  # D(f(x)) 
    

    # We firstly obtain the B'_h(\hat{y})，and then we could easily calculate the B'_h(\hat{y}|y). 
    y_h = model(user_embs_h10, ad_embs_h10, indicator_h10, 'original', is_training=False, label=labels_h10) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        y_h_0_list, y_h_1_list, label_list = [], [], [] 
        for _ in range(10): 
            (y_h_0, y_h_1), label = sess.run(y_h) 

            y_h_0_list.append(y_h_0) 
            y_h_1_list.append(y_h_1) 
            label_list.append(label) 

    y_h_0_stat = np.concatenate(y_h_0_list, 0).mean(0) # B'_h(\hat(y)|y=0) 
    y_h_1_stat = np.concatenate(y_h_1_list, 0).mean(0) # B'_h(\hat(y)|y=1) 
    y_h_stat   = np.stack([y_h_0_stat, y_h_1_stat]).T  # B'_h(\hat(y)|y), i.e., Confusion Matrix 

    label_stat = np.concatenate(label_list).mean(0) 
    label_stat = np.array([1-label_stat, label_stat])
    label_stat = np.expand_dims(label_stat, 1)  # B'_h(y) 

    # The esimated expectation of label distribution. 
    y_esm = DSC(y_h_stat, y_c_stat, 1.0, label_stat, 2) 

    ### Step 2: We finetune our production CVR prediction model based on the historical data and B_h(y) following Equation 6 in the paper. 
    y_t = model(user_embs_his, ad_embs_his, indicator_his, 'transblock', is_training=True) 

    labels = tf.stack([1-label, label], 1) 
    labels = tf.cast(labels, 'float32') 

    losses = -tf.reduce_mean(tf.log(y_t) * labels, 0) 
    (neg_loss, pos_loss) = tf.split(losses, 2, axis=0) 

    # Equation 6 in the paper 
    loss = y_esm[1][0]/label_stat[1][0] * pos_loss + neg_loss

    global_step = tf.get_variable(
        'global_step',
        [], 
        initializer=tf.constant_initializer(0), 
        trainable=False, 
        dtype='int32'
    ) 

    # We optimize the original model and TransBlock by different learning rate. 
    orginal_params = [val for val in tf.global_variables() if 'transblock' not in val.name] 
    optimizer_s = tf.train.AdamOptimizer(
        learning_rate=learning_rate_small, 
        beta1=0.9, 
        beta2=0.999, 
        epsilon=1e-8 
    ).minimize(loss, var_list=orginal_params, global_step=global_step) 

    transblock_params = [val for val in tf.global_variables() if 'transblock' in val.name] 
    optimizer_l = tf.train.AdamOptimizer(
        learning_rate=learning_rate_large, 
        beta1=0.9, 
        beta2=0.999, 
        epsilon=1e-8 
    ).minimize(loss, var_list=transblock_params, global_step=global_step) 


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        for _ in range(10): 
            _loss, _, _, _global_step = sess.run([loss, optimizer_s, optimizer_l, global_step]) 
            print('global_step: {}, loss: {}'.format(_global_step, _loss)) 
        print('Training Finished!') 

if __name__ == '__main__': 
    train() 
