import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

keep_rate = 0.8


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

# much_data = np.load('./segmented_npy/1a41350d4bbd74b7e0e28239cefa84c2.npy')
# print(much_data.shape)
# x_data = much_data[0][0]
# print(x_data.shape)
# y_data = much_data[0][1]
# print(y_data)
#

def predict_neural_network(x):
    prediction = convolutional_neural_network(x)

    much_data = np.load('patient.npy')
    x_data = much_data[0]


    with tf.Session() as sess:

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model_1.ckpt" )

        result = sess.run(prediction, feed_dict={x: x_data})

    return result

def predict_data():
    x = tf.placeholder('float')
    result = predict_neural_network(x)
    return np.argmax(result[0])