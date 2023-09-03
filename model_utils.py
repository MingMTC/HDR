#_*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
import pdb 

class Model(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplemented

class BN(Model):
    def __init__(self, name, center=True, scale=True, epsilon=1e-3, momentum=0.9):
        super(BN, self).__init__(name)
        self.bn_layer = tf.layers.BatchNormalization(
            name=name,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
        )
    def __call__(self, input, is_training=True):
        return self.bn_layer(input, training=is_training)


class Act(Model): 
    def __init__(self, name): 
        super(Act, self).__init__(name) 
        self.name = name 

    @staticmethod
    def _identity(x):
        return x

    @staticmethod
    def _relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def _softmax(x):
        return tf.nn.softmax(x)

    @staticmethod
    def _sigmoid(x):
        return tf.nn.sigmoid(x)

    @staticmethod
    def _swish(x):
        return tf.nn.sigmoid(x) * x

    @staticmethod
    def _elu(x):
        return tf.nn.elu(x)

    def __call__(self, input):
        return getattr(self, '_'+self.name)(input)

class PRelu(Model):
    def __init__(self, name, units):
        super(PRelu, self).__init__(name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.alpha = tf.get_variable(
                'prelu_alpha',
                shape=(units,),
                initializer=tf.constant_initializer(-0.25),
            )

    def __call__(self, input):
        a = tf.maximum(0.0, input)
        # print(a.graph, self.alpha.graph)
        out = a + self.alpha * tf.minimum(0.0, input)
        return out

class Dice(Model):
    def __init__(self, name, units):
        super(Dice, self).__init__(name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.dice_bn = BN(name='dice_bn', center=False, scale=False, epsilon=1e-4, momentum=0.99)
            self.dice_gamma = tf.get_variable(
                'dice_gamma',
                shape=(1, units),
                initializer=tf.constant_initializer(-0.25)
            )

    def __call__(self, input, is_training=True):
        out = self.dice_bn(input, is_training)
        logits = tf.nn.sigmoid(out)
        out = tf.multiply(self.dice_gamma, (1.0 - logits) * input) + logits * input
        return out


class Activation(object):
    def __init__(self, act_name, output_size=None):
        self.act_name = act_name
        self.output_size = output_size
        if not act_name:
            self.act = Act('identity')
        elif act_name.lower() == 'prelu':
            self.act = PRelu('prelu', self.output_size)
        elif act_name.lower() == 'dice':
            self.act = Dice('dice', self.output_size)
        else:
            self.act = Act(act_name)

    def __call__(self, input, is_training=True):
        if not self.act.name in ['dice']:
            return self.act(input)
        else:
            return self.act(input, is_training)


class Dense(Model):
    def __init__(self, name, input_size, output_size, act_type, use_bn=False, use_bias=True):
        super(Dense, self).__init__(name)
        self.name=name 
        self.input_size = input_size
        self.output_size = output_size
        self.act_type = act_type
        self.use_bn = use_bn
        self.use_bias = use_bias

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            weight_value = np.random.randn(self.output_size, self.input_size).astype(
                np.float32
            ) / np.sqrt(self.input_size)
            weight_value = np.transpose(weight_value)
            self.dense_layer = tf.layers.Dense(
                self.output_size,
                name=self.name+'_fc',
                use_bias=self.use_bias,
                activation=None,
                kernel_initializer=tf.initializers.constant(weight_value),
                bias_initializer=tf.initializers.constant(0.1)
            )
            self.bn = BN(name='bn')
            self.act = Activation(
                act_type,
                output_size=self.output_size if act_type.lower() in ['prelu', 'dice'] else None
            )

    def __call__(self, input, is_training=True):
        out = self.dense_layer(input)
        if self.use_bn and self.act_type.lower() != 'dice':
            out = self.bn(out, is_training)
        out = self.act(out, is_training)
        return out


class MLP(Model):
    def __init__(self, name, input_size, network_args, act_type=None, use_bn=False, use_bias=True):
        super(MLP, self).__init__(name) 
        assert len(network_args) > 0, 'empty network_args!' 

        self.input_size   = input_size 
        self.network_args = network_args 
        self.act_type     = act_type 
        self.use_bn       = use_bn 
        self.use_bias     = use_bias 

        self.layers = [] 
        units_in = self.input_size
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for idx, units in enumerate(self.network_args, 1):
                self.layers.append(
                    Dense(
                        name='_'.join(map(str, [name, 'fc', idx])), ## 'name_fc_1', 'name_fc_2', ...
                        input_size=units_in,
                        output_size=units,
                        act_type=self.act_type, 
                        use_bn=self.use_bn,
                        use_bias=self.use_bias 
                    )
                )
                units_in = units 
    
    def __call__(self, input, is_training=True):
        out = input
        for layer in self.layers:
            out = layer(out, is_training)
        return out
