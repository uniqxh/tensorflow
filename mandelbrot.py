#!/usr/bin/python

import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.ndimage as nd

def show(a):
    b = (6.28*a/20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10+20*np.cos(b),
                          30+50*np.sin(b),
                          155-80*np.cos(b)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    im = Image.fromarray(a)
    im.show()
    im.save('mandelbrot.jpg')

sess = tf.InteractiveSession()
y,x = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
z = x+1j*y
xs = tf.constant(z.astype('complex64'))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, 'float32'))

tf.global_variables_initializer().run()

zs_ = zs*zs + xs
f = tf.complex_abs(zs_) < 4
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(f, 'float32')))

for i in range(200):
    step.run()

show(ns.eval())
