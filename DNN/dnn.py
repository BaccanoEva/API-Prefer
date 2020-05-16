import os
import tensorflow as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
from data import generate_data

batch_size= 500
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data warehouse, waiting to be used
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train)

x_train,y_train,x_test,y_test = generate_data.generate_train()

# 1、define placeholder for input
xs = tf.placeholder(dtype=tf.float32, shape=(None, 600))
ys = tf.placeholder(dtype=tf.float32, shape=(None, 4))
in_image = tf.reshape(xs, [-1, 3, 200,1])  # resize the input

# construct network begin #
# conv1
h_conv1 = tf.layers.conv2d(
    inputs=in_image,
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)  # output size: [n_samples,28,28,32]

# maxpooling1
h_pool1 = tf.layers.max_pooling2d(
    h_conv1,
    pool_size=(3, 10),
    strides=2,
    padding='same')  # output size: [n_samples,14,14,32]

# conv2
h_conv2 = tf.layers.conv2d(
    inputs=h_pool1,
    filters=64,
    kernel_size=(1, 1),
    padding='same',
    activation=tf.nn.relu)  # output size: [n_samples,14,14,64]

# maxpooling2
h_pool2 = tf.layers.max_pooling2d(
    h_conv2,
    pool_size=(1, 2),
    strides=2,
    padding='same')  # output size: [n_samples,7,7,64]

# fc1
# h_pool2_flat = tf.layers.Flatten(h_pool2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 10 * 64])  # from [n_samples,7,7,64] to [n_samples,7*7*64]
h_fc1 = tf.layers.dense(h_pool2_flat, units=256, activation=tf.nn.relu)  # from [n_samples,7*7*64] to [n_samples,1024]
print(h_fc1.shape)
# fc2
prediction = tf.layers.dense(h_fc1, units=4, activation=tf.nn.softmax)
print(prediction.shape)

# construct network end #

# define the target
cross_entropy = tf.losses.softmax_cross_entropy(ys, prediction)  # ys size [batch_size, 10]

# define the optimizer(how to get to the target)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# very important
init = tf.global_variables_initializer()
sess.run(init)

def compute_accuracy(v_xs, v_ys):
    global prediction
    pred = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(v_ys, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

for i in range(500):
    data_size = 44960
    start = i * batch_size % data_size
    end = min(start + batch_size,data_size)
    batch_xs, batch_ys = x_train[start:end],y_train[start:end]
    print(len(batch_ys),batch_ys[0])

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    # use test data to compute accuracy
    if i % 50 == 0:
        print(compute_accuracy(x_test, y_test))