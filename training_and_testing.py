from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
#in simple words thry are the X(inputs)
train_features = mnist.train.images
test_features = mnist.test.images
#in simple words they are y(the correct label)
train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.zeros([n_classes]))

#defining the features
features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])
#use of None is that we cam implement a batch_size in its place during session

training_feed_dict = {features: train_features, labels: train_labels}
validation_feed_dict = {features: valid_features, labels: valid_labels}
testing_feed_dict = {features: test_features, labels: test_labels}

#doing yhat = xw + b
logits = tf.add(tf.matmul(features,weights),bias)

#calculating the error using cross entropy
#E = -1/m sumof(y*ln(p))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

#the alternative to above is
'''probability = tf.nn.softmax(logits)
cost = -tf.reduce_sum(labels* tf.log(probability),axis=1)'''

#calculating the loss
loss = tf.reduce_mean(cost)

#initializing the variables
init = tf.global_variables_initializer()

#setting the the parameters
epoch = 5
batch_size = 100
learning_rate = 0.2

#updating the weights using the Gradient Descent method
#wi = wi - (learing_rate*(dE/dWi))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches

with tf.Session() as sess:
    sess.run(init)
    sess.run(loss,feed_dict=training_feed_dict)
    sess.run(loss,feed_dict=validation_feed_dict)
    sess.run(loss,feed_dict=testing_feed_dict)
    '''Calculate accuracy for test dataset
    test_accuracy = sess.run(
    accuracy,
    feed_dict={features: test_features, labels: test_labels})''' # this is equivalentsess.run(accuracy, feed_dict=testing_feed_dict)

print('Test Accuracy: {}'.format(test_accuracy))

#calculating the accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch):
        for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
            #sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
            _, l = sess.run(
                [optimizer, loss],
            feed_dict={features: batch_features, labels: batch_labels})
            accuracy_training = sess.run(accuracy, feed_dict=training_feed_dict)
            print(accuracy_training)
            accuracy_validation = sess.run(accuracy, feed_dict=validation_feed_dict)
            print(accuracy_validation)
            accuracy_testing = sess.run(accuracy, feed_dict=testing_feed_dict)
            print(acuracy_testing)
