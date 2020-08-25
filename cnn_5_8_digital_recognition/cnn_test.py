# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import sys
from PIL import Image#pillow(PIL)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)#创建常量
  return tf.Variable(initial)

#权重初始化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#卷积操作

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#d第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#训练和评估模型
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

ckpt_dir = "./ckpt_dir"

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
    else:
        raise FileNotFoundError("未找到模型")#raise 引发异常

    # image_path="D:\\1.png"
    # image_path="/home/mnist/mnist03/test_data/"+sys.argv[1]+".png"
    image_path=sys.argv[1]
    print(image_path)
    img = Image.open(image_path).convert('L')#灰度图(L)
    print(img.size)
    img = img.resize((28,28))
    img_shape = np.reshape(img, 784)
    real_x = np.array([1-img_shape])# 0-255 uint8   8位无符号整数，取值：[0, 255] 如果采用1-大数变成小数
    y = sess.run(y_conv, feed_dict={x: real_x,keep_prob: 1.0}) #y类似一个二维表，因为只有一张图片所以只有一行，y[0]包含10个值，

    print('Predict digit', np.argmax(y[0]))#找出最大的值

