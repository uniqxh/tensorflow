#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
from PIL import Image
import sys,os

def w_v(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def b_v(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", [None, 784])

w_conv1 = w_v([5, 5, 1, 32])
b_conv1 = b_v([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

w_conv2 = w_v([5, 5, 32, 64])
b_conv2 = b_v([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

w_fc1 = w_v([7*7*64, 1024])
b_fc1 = b_v([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
k_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, k_prob)

w_fc2 = w_v([1024, 10])
b_fc2 = b_v([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)


y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, './trainData/train')

print 'read train data successful.'
while True:
    path = raw_input("请输入要识别的图像名称：")
    print 'filename is : ', path
    if os.path.exists(path) == False:
        print path,' 不存在'
        continue
    im = Image.open(path)
    if im.size != (28, 28):
        im = im.resize((28, 28), Image.ANTIALIAS)
    if im.mode != 'L':
        im = im.convert('L')
    dt = list(im.getdata())
    dt = map(lambda i: (255 - dt[i])*1.0/255.0, range(len(dt)))
    re = y_conv.eval(feed_dict={x: [dt], k_prob: 1.0})
    nm = tf.argmax(re, 1)
    #print re
    print '识别结果：',nm.eval()




