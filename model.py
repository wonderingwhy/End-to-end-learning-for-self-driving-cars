'''
You are going to implement the CNN model from paper 'End to End Learning for Self-Driving Cars'.
Write the model below.
'''

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
keep_prob = tf.placeholder(tf.float32)

# 输入层
tf_X = tf.placeholder(tf.float32, [None, 66, 200, 3])
tf_Y = tf.placeholder(tf.float32, [None, 1])

x_image = tf_X
# 归一化
# batch_mean, batch_var = tf.nn.moments(tf_X, [0, 1, 2], keep_dims = True)
# shift = tf.Variable(tf.zeros([3]))
# scale = tf.Variable(tf.ones([3]))
# epsilon = 1e-3
# BN_out = tf.nn.batch_normalization(tf_X, batch_mean, batch_var, shift, scale, epsilon)
# print(BN_out.shape)

# 卷积层1
conv_filter_w1 = tf.Variable(tf.random_normal([5, 5, 3, 24], stddev = 0.1))
# conv_filter_b1 = tf.Variable(tf.random_normal([24], stddev = 0.1))
conv_filter_b1 = tf.Variable(tf.constant(0.1, shape = [24]))
conv_out1 = tf.nn.conv2d(x_image, conv_filter_w1, strides = [1, 2, 2, 1], padding = 'VALID') + conv_filter_b1;

# 归一化1
batch_mean, batch_var = tf.nn.moments(conv_out1, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([24]))
scale = tf.Variable(tf.ones([24]))
epsilon = 1e-3
BN_out1 = tf.nn.batch_normalization(conv_out1, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数1
relu_feature_maps1 = tf.nn.relu(BN_out1)
# print(relu_feature_maps1.shape)

# 卷积层2
conv_filter_w2 = tf.Variable(tf.random_normal([5, 5, 24, 36], stddev = 0.1))
# conv_filter_b2 = tf.Variable(tf.random_normal([36], stddev = 0.1))
conv_filter_b2 = tf.Variable(tf.constant(0.1, shape = [36]))
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2, strides = [1, 2, 2, 1], padding = 'VALID') + conv_filter_b2

# 归一化2
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([36]))
scale = tf.Variable(tf.ones([36]))
epsilon = 1e-3
BN_out2 = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数2
relu_feature_maps2 = tf.nn.relu(BN_out2)
# print(relu_feature_maps2.shape)

# 卷积层3
conv_filter_w3 = tf.Variable(tf.random_normal([5, 5, 36, 48], stddev = 0.1))
# conv_filter_b3 = tf.Variable(tf.random_normal([48], stddev = 0.1))
conv_filter_b3 = tf.Variable(tf.constant(0.1, shape = [48]))
conv_out3 = tf.nn.conv2d(relu_feature_maps2, conv_filter_w3, strides = [1, 2, 2, 1], padding = 'VALID') + conv_filter_b3

# 归一化3
batch_mean, batch_var = tf.nn.moments(conv_out3, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([48]))
scale = tf.Variable(tf.ones([48]))
epsilon = 1e-3
BN_out3 = tf.nn.batch_normalization(conv_out3, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数3
relu_feature_maps3 = tf.nn.relu(BN_out3)
# print(relu_feature_maps3.shape)

# 卷积层4
conv_filter_w4 = tf.Variable(tf.random_normal([3, 3, 48, 64], stddev = 0.1))
# conv_filter_b4 = tf.Variable(tf.random_normal([64], stddev = 0.1))
conv_filter_b4 = tf.Variable(tf.constant(0.1, shape = [64]))
conv_out4 = tf.nn.conv2d(relu_feature_maps3, conv_filter_w4, strides = [1, 1, 1, 1], padding = 'VALID') + conv_filter_b4

# 归一化4
batch_mean, batch_var = tf.nn.moments(conv_out4, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([64]))
scale = tf.Variable(tf.ones([64]))
epsilon = 1e-3
BN_out4 = tf.nn.batch_normalization(conv_out4, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数4
relu_feature_maps4 = tf.nn.relu(BN_out4)
# print(relu_feature_maps4.shape)

# 卷积层5
conv_filter_w5 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev = 0.1))
# conv_filter_b5 = tf.Variable(tf.random_normal([64], stddev = 0.1))
conv_filter_b5 = tf.Variable(tf.constant(0.1, shape = [64]))
conv_out5 = tf.nn.conv2d(relu_feature_maps4, conv_filter_w5, strides = [1, 1, 1, 1], padding = 'VALID') + conv_filter_b5

# 归一化5
batch_mean, batch_var = tf.nn.moments(conv_out5, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([64]))
scale = tf.Variable(tf.ones([64]))
epsilon = 1e-3
BN_out5 = tf.nn.batch_normalization(conv_out5, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数5
relu_feature_maps5 = tf.nn.relu(BN_out5)
# print(relu_feature_maps5.shape)

# flatten
flat = tf.reshape(relu_feature_maps5, [-1, 1152])
# print(flat)

# 全连接层1
fc_w1 = tf.Variable(tf.random_normal([1152, 100], stddev = 0.1))
# fc_b1 =  tf.Variable(tf.random_normal([100], stddev = 0.1))
fc_b1 = tf.Variable(tf.constant(0.1, shape = [100]))
fc_y1 = tf.matmul(flat, fc_w1) + fc_b1

# 归一化6
batch_mean, batch_var = tf.nn.moments(fc_y1, [0], keep_dims = True)
shift = tf.Variable(tf.zeros([1]))
scale = tf.Variable(tf.ones([1]))
epsilon = 1e-3
BN_out6 = tf.nn.batch_normalization(fc_y1, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数6
fc_out1 = tf.nn.relu(fc_y1)
fc_drop1 = tf.nn.dropout(fc_out1, keep_prob)

# 全连接层2
fc_w2 = tf.Variable(tf.random_normal([100, 50], stddev = 0.1))
# fc_b2 =  tf.Variable(tf.random_normal([50], stddev = 0.1))
fc_b2 = tf.Variable(tf.constant(0.1, shape = [50]))
fc_y2 = tf.matmul(fc_out1, fc_w2) + fc_b2

# 归一化7
batch_mean, batch_var = tf.nn.moments(fc_y2, [0], keep_dims = True)
shift = tf.Variable(tf.zeros([1]))
scale = tf.Variable(tf.ones([1]))
epsilon = 1e-3
BN_out7 = tf.nn.batch_normalization(fc_y2, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数7
fc_out2 = tf.nn.relu(BN_out7)
fc_drop2 = tf.nn.dropout(fc_out2, keep_prob)

# 全连接层3
fc_w3 = tf.Variable(tf.random_normal([50, 10], stddev = 0.1))
# fc_b3 =  tf.Variable(tf.random_normal([10], stddev = 0.1))
fc_b3 = tf.Variable(tf.constant(0.1, shape = [10]))
fc_y3 = tf.matmul(fc_out2, fc_w3) + fc_b3

# 归一化8
batch_mean, batch_var = tf.nn.moments(fc_y3, [0], keep_dims = True)
shift = tf.Variable(tf.zeros([1]))
scale = tf.Variable(tf.ones([1]))
epsilon = 1e-3
BN_out8 = tf.nn.batch_normalization(fc_y3, batch_mean, batch_var, shift, scale, epsilon)

# 激活函数8
fc_out3 = tf.nn.relu(BN_out8)
fc_drop3 = tf.nn.dropout(fc_out3, keep_prob)

# 输出层
out_w1 = tf.Variable(tf.random_normal([10, 1], stddev = 0.1))
# out_b1 = tf.Variable(tf.random_normal([1], stddev = 0.1))
out_b1 = tf.Variable(tf.constant(0.1, shape = [1]))
out_y1 = tf.matmul(fc_out3, out_w1) + out_b1

# 归一化9
batch_mean, batch_var = tf.nn.moments(out_y1, [0], keep_dims = True)
shift = tf.Variable(tf.zeros([1]))
scale = tf.Variable(tf.ones([1]))
epsilon = 1e-3
BN_out9 = tf.nn.batch_normalization(out_y1, batch_mean, batch_var, shift, scale, epsilon)

# 非线性映射
y = tf.multiply(tf.atan(BN_out9), 2);
# print(y)
