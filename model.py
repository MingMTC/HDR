#_*_ coding:utf-8 _*_
import tensorflow as tf

from model_utils import Model, MLP, BN

import pdb 


class CVRModel(Model):
    # A toy CVR prediction model 
    def __init__(self, name, input_size, layer_args):
        super(CVRModel, self).__init__(name)

        self.eps = 0.00000001
        # original CVR prediction model 
        self.bn = BN( 
                '_'.join([name, 'bn']), 
                center=True, 
                scale=True, 
                epsilon=1e-3, 
                momentum=0.9 
            )

        self.org_model = [
            MLP( 
                '_'.join([name, 'mlp']), 
                input_size, 
                layer_args[:-1], 
                act_type='dice', 
                use_bn=False, 
                use_bias=True 
            ), 
            MLP( 
                '_'.join([name, 'proj_layer']), 
                layer_args[-2], 
                layer_args[-1:], 
                act_type='softmax', 
                use_bn=False, 
                use_bias=True 
            ) 
        ] 

        # transblock layer 
        self.transblock = MLP( 
                'transblock', 
                layer_args[-2], 
                layer_args[-1:], 
                act_type='identity', 
                use_bn=False, 
                use_bias=True 
            )
    
    def __call__(self, user_embs, ad_embs, indicator, mode, is_training=True, label=None, tb_act='sigmoid_boost'): 
        """
        @params:
            user_embs: user embeddings, shape of [B1, E1]
            ad_embs: ad embeddings, shape of [B2, E2]
            indicator: shape of [B2], with values between [0, B1 - 1]
            mode: original/transblock 
            is_training: for batchnorm 
        """
        if indicator is not None:
            user_embs = tf.gather(user_embs, indices=indicator, axis=0)
        embs = tf.concat([user_embs, ad_embs], axis=1) 

        embs_bn = self.bn(embs, is_training=is_training) 
        h_x     = self.org_model[0](embs_bn, is_training) 
        p       = self.org_model[1](h_x, is_training) 

        if mode == 'original': # original CVR prediction model 
            output = p + self.eps 
        elif mode == 'transblock': # CVR prediction model + TransBlock 
            # split the original output 
            p_0, p_1 = tf.split(p, 2, axis=1) 

            # transition probabilities from transblock 
            wb = self.transblock(h_x, is_training) 
            w_x, b_x = tf.split(wb, 2, axis=1) 

            if tb_act=='sigmoid_boost': 
                w_x = tf.sigmoid(w_x) * 2 - 1.0 
            elif tb_act=='sigmoid': 
                w_x = tf.sigmoid(w_x) 
            elif tb_act=='NULL': 
                w_x = w_x 
            else: 
                assert False, 'Errors in activation function.' 

            # Eq.2 in ATA 
            p_t_1 = p_1 + (w_x * p_0 + b_x) 
            p_t_0 = p_0 - (w_x * p_0 + b_x) 

            p_t = tf.concat([p_t_0, p_t_1], 1) 

            # guarantee the range of output 
            p_t = tf.clip_by_value(p_t, 0.0, 1.0) 

            output = p_t + self.eps 

        # split the predictive scores by label for D(f(x')|y') 
        if label is not None: 
            output = tf.dynamic_partition(output, label, 2) 

            return output, label 

        return output 
