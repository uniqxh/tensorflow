#!/usr/bin/python
import tensorflow as tf
import numpy as np
from PIL import Image
from cStringIO import StringIO
import images2gif
#from IPython.display import clear_output, Image, display
images = []
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
    a = (a-rng[0])/float(rng[1] - rng[0])*255
    a = np.uint8(np.clip(a, 0 , 255))
    f = StringIO()
    m = Image.fromarray(a)
    #clear_output(wait = True)
    images.append(m)

sess = tf.InteractiveSession()

def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1,1,1,1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

N = 500
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

for n in range(40):
    a,b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])

eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

U_ = U + eps*Ut
Ut_ = Ut + eps*(laplace(U) - damping*Ut)

step = tf.group(U.assign(U_), Ut.assign(Ut_))

tf.global_variables_initializer().run()

for i in range(10):
    step.run({eps: 0.1, damping: 0.1})
    DisplayArray(U.eval(), rng=[-0.1, 0.1])

size = (500,500)
for im in images:
        im.thumbnail(size, Image.ANTIALIAS)
images2gif.writeGif('test.gif', images, duration=0.1, subRectangles=False)
