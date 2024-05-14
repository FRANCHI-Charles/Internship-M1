import tensorflow as tf
import numpy as np

# Tensors in tf are immutable

tf.constant([[1,2,3],[4,5,6]])
tf.ones((3,3), dtype= tf.float32)
tf.zeros((4,5,6,7))
print(tf.random.normal((1,4,5)))
a = tf.range(10)
tf.reshape(a, (-1,2))
a.numpy()

tf.constant(np.array([4,5,6]))

try:
    a[0] = 4
except:
    print("nope")

b = tf.Variable([4,5,6])

try:
    b.assign([7,8])
except:
    print("not the good shape!")