import tensorflow as tf
import numpy as np 
import pdb


'''
We provide a toy tf-style pseudocode to illustrate the details of TransBlock and Distribution Shift Correction (DSC). 
'''

eps = 0.00000001 

def CVR_model_toy(user, ad, mode, tb_act='sigmoid_boost'): 
    ''' 
    user: user feature, [batch_size, user_feasize] 
    ad:   ad feature, [batch_size, ad_feasize] 
    ''' 
    # original model 
    fea = tf.concat([user, ad], axis=1) 
    bn_out = bn(fea, is_training, scope='bn') 
    fc1  = tf.layers.dense(bn_out, 500, activation='relu', name='fc1') 
    fc2  = tf.layers.dense(fc1,    200, activation='relu', name='fc2') 
    fc3  = tf.layers.dense(fc2,    100, activation='relu', name='fc3') 
    fc4  = tf.layers.dense(fc3,    2,   activation='',     name='fc4') 

    # output from orginal model 
    prop = tf.nn.softmax(fc6) 

    if mode == 'pred_score': 
        prop = prop + eps 
    elif mode == 'TransBlock': ### TransBlock 
        p0_org = tf.slice(prop, [0, 0], [-1, 1]) 
        p1_org = tf.slice(prop, [0, 1], [-1, 1]) 

        fc_trans = tf.layers.dense(fc3, 2, activation='', name='param_trans') 

        if tb_act=='sigmoid_boost': 
            trans_ratio = tf.sigmoid(fc_trans) * 2 - 1.0 
        elif tb_act=='sigmoid': 
            trans_ratio = tf.sigmoid(fc_trans) 
        elif tb_act=='NULL': 
            trans_ratio = fc_trans 
        else: 
            assert False, 'Errors in activation function.' 

        v = tf.slice(trans_ratio, [0, 0], [-1, 1]) 
        u = tf.slice(trans_ratio, [0, 1], [-1, 1]) 

        res_val = v * p0_org - u * p1_org 
        p1 = p1_org + res_val 
        p0 = p0_org - res_val 

        # ensure the value in [0.0, 1.0] 
        p1 = tf.minimum(tf.maximum(p1, 0.0), 1.0) 
        p0 = tf.minimum(tf.maximum(p0, 0.0), 1.0) 

        prop = tf.concat([p0, p1], axis=1) + eps 

    return prop 

def DSC(C, q, lamda, h, dim): 
    '''
    We compute the B_h(y) as the Equation 14 (Appendix) which is the closed-form solution of Equation 7.

    q: M'_{\hat{y}}, predition scores from f_\Theta 
    C: M'_{\hat{y}|y}, confusion matrix from historical Data 
    lamda: the weight for regularization 
    h: label distribution from Historical Data 
    dim: the size of confusion matrix. It is notable that we could compute importance weight for each instance. 
    ''' 
    I_matrix = np.matrix(np.identity(dim)) 
    return (C.T * C + lamda * I_matrix).I * (C.T * q + lamda * h) 

def Finetune(user, ad, 
             user_h, ad_h, label_h, 
             user_h_10h, ad_h_10h, label_h_10h, 
             C, lamda=1): 
    '''
    user, ad: the features from first 10 hour of this day 
    user_h, ad_h: the features in retrieved historical promotion data 
    label_h, label_h_10h: the label of historical data in whole day and the first 10 hours 
    C: M'_{\hat{y}|y}, confusion matrix from historical Data (the first 10 hours) 
    '''
    ### obtaining for M'_{\hat{y}}, we call it q 
    his_prop = CVR_model_toy(user_h_10h, ad_h_10h, mode='pred_score') 
    q = tf.reduce_mean(his_prop, axis=1) 

    ### Distribution shift correction, we omit the process that converts tf-tensor to np-tensor 
    p = DSC(C, q, lamda, label_h_10h, dim=2)   # B_h(y) 

    ### fine-tuning 
    prop = CVR_model_toy(user, ad, mode='TransBlock') 

    p_10h = tf.reduce_mean(label_h_10h, axis=1) # B'_h(y) 
    loss = -tf.reduce_mean(label_h * (p/p_10h) * tf.log(prop)) 

if __name__ == '__main__':
    Finetune(user, ad, 
             user_h, ad_h, label_h, 
             user_h_10h, ad_h_10h, label_h_10h, 
             C, lamda=1) 
