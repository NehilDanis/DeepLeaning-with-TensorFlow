import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion', one_hot=True)

# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)
train_y = data.train.labels
test_y = data.test.labels

n_classes = 10
batch_size = 128  # the number of samples in each batch
n_epochs = 200  # the number of feed forward + back prop
# MNIST data input (img shape: 28*28)
n_input = 28

x = tf.placeholder('float', [None, n_input, n_input, 1])  # 28x28 --> input
y = tf.placeholder('float', [None, n_classes])  # --> estimated output


# xavier initialization is used. Since it keeps the variance of all random weights finite,
# (gaussian distribution)
#  the weights will never be either too large or too small.
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),  # in the first conv. layer there is 32 filters
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),  # there will be 3 x 3 x 32 64 filters, since the output of the first kayer is 32
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),  # after 3 conv and max pool layer out image is downsampled to 4 x 4 x 1
    # the second 128 shows the number of nodes in the fully connected layer.
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}



def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def conv_net_model(data, weights, biases):
    '''
    :param data: one input sample, 28 x 28 x 1
    :param weights: python dictionary --> keeps all the weights for the cnn.
    :param biases: python dictionary --> keeps all the biases for the cnn.
    :return: the output of this function is the prediction which is made by out convolutional network.
    '''

    conv1 = conv2d(data, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    # Fully Connected Layer
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]]) # here all the tensors from conv layer 3
    # are added into one array.
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  # (input_sample x weights) + biases
    fc1 = tf.nn.relu(fc1)  # activation function is applied to the fcl.

    # Output, class prediction
    # we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


def train_conv_net(x, y):
    prediction = conv_net_model(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    # optionally you can add learning rate to the Adam Optimizer. But by default it is set to 0.001.
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, n_epochs):
            epoch_loss = 0
            for _ in range(0, len(train_X)//batch_size):
                epoch_x = train_X[_ * batch_size:min((_ + 1) * batch_size, len(train_X))]
                epoch_y = train_y[_ * batch_size:min((_ + 1) * batch_size, len(train_y))]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # argmax finds the maximum value in the array. Since it is a one hot vector
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # reduce_mean computes the mean of elements across dimensions of a tensor.
        print('Accuracy:', accuracy.eval({x: data.test.images, y: data.test.labels}))


train_conv_net(train_X, train_y)


