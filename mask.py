# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:00:05 2017

@author: xuenene
"""
import tensorflow as tf
import model
import load_data
import scipy.misc
import numpy as np  
import cv2

saver = tf.train.Saver()

batch_size = 1
image_num = 45568
def average_map(feature_maps):
    length = len(feature_maps[0][0][0])
    for i in range(1, length):
        feature_maps[:,:,:,0] += feature_maps[:,:,:,i]
    feature_maps[:,:,:,0] /= length
    return feature_maps[:,:,:,:1];

def upscale(average_map, height, width):
    average_map = scipy.misc.imresize(average_map, [height, width])
    #print(average_map.shape)
    return average_map

def point_mul(mask, ave_feature_map):
    #print(ave_feature_map)
    if mask.shape != ave_feature_map.shape :
        print("shape is not same!")
    else:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                ave_feature_map[i, j] *= mask[i, j]
    #print(ave_feature_map)
    return ave_feature_map

feature_maps_path = "./feature_maps/"
mask_path = "./mask/"
image_path = "./image/"
def save_mask(mask, name):
    img = scipy.misc.imresize(mask, [66, 200])
    img = img / np.max(img)
    cv2.imwrite(mask_path + name, img * 255)
    return img

def save_map(feature_map, name):
    img = scipy.misc.imresize(feature_map, [66, 200])
    img = img / np.max(img)
    cv2.imwrite(feature_maps_path + name, img * 255)
    return img

def rise_dimension(matrix):
    ret = np.zeros([1, len(matrix[:, 0]), len(matrix[0, :]), 1], np.float32)
    ret[0,:,:,0] = matrix
    return ret

with tf.Session() as sess:
    saver.restore(sess, "save_model/model.ckpt")
    for t in range(int(image_num / batch_size)):
        xs, ys = load_data.LoadTrainBatch(batch_size)
        
        feature_maps5 = sess.run(model.relu_feature_maps5, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map5 = average_map(feature_maps5)[0,:,:,0]
        save_map(ave_map5, str(t) + "_map5.jpg")
        
        mask5 = tf.nn.conv2d_transpose(rise_dimension(ave_map5), tf.constant(1.0, shape = [3, 3, 1, 1]), [batch_size, 3, 20, 1], strides = [1, 1, 1, 1], padding = 'VALID').eval()[0,:,:,0]
        save_mask(mask5, str(t) + "_mask5.jpg")
        
        feature_maps4 = sess.run(model.relu_feature_maps4, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map4 = average_map(feature_maps4)[0,:,:,0]
        save_map(ave_map4, str(t) + "_map4.jpg")
        mask4 = tf.nn.conv2d_transpose(rise_dimension(point_mul(mask5, ave_map4)), tf.constant(1.0, shape = [3, 3, 1, 1]), [batch_size, 5, 22, 1], strides = [1, 1, 1, 1], padding = 'VALID').eval()[0,:,:,0]
        save_mask(mask4, str(t) + "_mask4.jpg")
        
        feature_maps3 = sess.run(model.relu_feature_maps3, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map3 = average_map(feature_maps3)[0,:,:,0]
        save_map(ave_map3, str(t) + "_map3.jpg")
        mask3 = tf.nn.conv2d_transpose(rise_dimension(point_mul(mask4, ave_map3)), tf.constant(1.0, shape = [5, 5, 1, 1]), [batch_size, 14, 47, 1], strides = [1, 2, 2, 1], padding = 'VALID').eval()[0,:,:,0] 
        save_mask(mask3, str(t) + "_mask3.jpg")
        
        feature_maps2 = sess.run(model.relu_feature_maps2, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map2 = average_map(feature_maps2)[0,:,:,0]
        save_map(ave_map2, str(t) + "_map2.jpg")
        mask2 = tf.nn.conv2d_transpose(rise_dimension(point_mul(mask3, ave_map2)), tf.constant(1.0, shape = [5, 5, 1, 1]), [batch_size, 31, 98, 1], strides = [1, 2, 2, 1], padding = 'VALID').eval()[0,:,:,0]  
        save_mask(mask2, str(t) + "_mask2.jpg")
        
        feature_maps1 = sess.run(model.relu_feature_maps1, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map1 = average_map(feature_maps1)[0,:,:,0]
        save_map(ave_map1, str(t) + "_map1.jpg")
        mask1 = tf.nn.conv2d_transpose(rise_dimension(point_mul(mask2, ave_map1)), tf.constant(1.0, shape = [5, 5, 1, 1]), [batch_size, 66, 200, 1], strides = [1, 2, 2, 1], padding = 'VALID').eval()[0,:,:,0]   
        mask = save_mask(mask1, str(t) + "_mask1.jpg")
        
        img = np.zeros([66, 200, 3], np.uint8)
        for j in range(len(xs[0][:,0,0])):
            for k in range(len(xs[0][0,:,0])):
                for l in range(len(xs[0][0,0,:])):
                    img[j, k, l] = int(xs[0][j, k, l] * 255.0 + 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        
        for j in range(len(img[:, 0, 0])):
            for k in range(len(img[0, :, 0])):
                #if(mask[j, k] > 0.2):
                    #mask[j, k] = pow(mask[j, k], 0.3)
                    img[j, k, 1] = max(img[j, k, 1], int(pow(mask[j, k], 0.4) * 255 + 0.5))
        cv2.imwrite(image_path + str(t) + '.jpg', img)
        print(str(t) + '.jpg saved')
"""
with tf.Session() as sess:
    saver.restore(sess, "save_model/model.ckpt")
    for t in range(int(image_num / batch_size)):
        xs, ys = load_data.LoadTrainBatch(batch_size)
        
        feature_maps5 = sess.run(model.relu_feature_maps5, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map5 = average_map(feature_maps5)[0,:,:,0]
        save_map(ave_map5, str(t) + "_map5.jpg")
        mask5 = upscale(ave_map5, 3, 20)
        save_mask(mask5, str(t) + "_mask5.jpg")
        
        feature_maps4 = sess.run(model.relu_feature_maps4, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map4 = average_map(feature_maps4)[0,:,:,0]
        save_map(ave_map4, str(t) + "_map4.jpg")
        mask4 = upscale(point_mul(mask5, ave_map4), 5, 22)
        save_mask(mask4, str(t) + "_mask4.jpg")
        
        feature_maps3 = sess.run(model.relu_feature_maps3, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map3 = average_map(feature_maps3)[0,:,:,0]
        save_map(ave_map3, str(t) + "_map3.jpg")
        mask3 = upscale(point_mul(mask4, ave_map3), 14, 47)
        save_mask(mask3, str(t) + "_mask3.jpg")
        
        feature_maps2 = sess.run(model.relu_feature_maps2, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map2 = average_map(feature_maps2)[0,:,:,0]
        save_map(ave_map2, str(t) + "_map2.jpg")
        mask2 = upscale(point_mul(mask3, ave_map2), 31, 98) 
        save_mask(mask2, str(t) + "_mask2.jpg")
        
        feature_maps1 = sess.run(model.relu_feature_maps1, \
                                 feed_dict = {model.tf_X: xs, model.tf_Y: ys, model.keep_prob: 0.8})
        ave_map1 = average_map(feature_maps1)[0,:,:,0]
        save_map(ave_map1, str(t) + "_map1.jpg")
        mask1 = upscale(point_mul(mask2, ave_map1), 66, 200)
        mask = save_mask(mask1, str(t) + "_mask1.jpg")
        
        img = np.zeros([66, 200, 3], np.uint8)
        for j in range(len(xs[0][:,0,0])):
            for k in range(len(xs[0][0,:,0])):
                for l in range(len(xs[0][0,0,:])):
                    img[j, k, l] = int(xs[0][j, k, l] * 255.0 + 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        
        for j in range(len(img[:, 0, 0])):
            for k in range(len(img[0, :, 0])):
                    img[j, k, 1] = max(img[j, k, 1], int(pow(mask[j, k], 0.5) * 255 + 0.5))
        cv2.imwrite(image_path + str(t) + '.jpg', img)
        print(str(t) + '.jpg saved')
"""