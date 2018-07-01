import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 16
SLICE_COUNT = 16

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               #                                  64 features
               'W_fc': tf.Variable(tf.random_normal([4*4*64, 1000])),
               'out': tf.Variable(tf.random_normal([1000, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1000])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z           reshape the image to the correct size
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    # ReLU = rectified linear units (negative values become 0)
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    # final grid dimensions times the number of channel
    fc = tf.reshape(conv2, [-1, 4 * 4 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(train_data, validation_data):
    x = tf.placeholder('float', [None, 256])
    y = tf.placeholder('float', [None, 2])

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(len(train_data.images)):
                total_runs += 1
                try:
                    X = train_data.images[i]
                    Y = train_data.labels[i]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    # print(str(e))

            print('Epoch', epoch+1, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # print('Accuracy:', accuracy.eval(
            # {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        # print('Accuracy:', accuracy.eval(
        # {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        print('fitment percent:', successful_runs/total_runs)

# Run this locally:
# train_neural_network(x)
