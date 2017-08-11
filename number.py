import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

def compute_accuracy(v_x,v_y):
    global prediction
    y_pre = sess.run(prediction,feed_dict={x:v_x,keep_prob:1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={x:v_x, y:v_y,keep_prob:1.0})
    return  result
def cov2d(x_data, W):
    return tf.nn.conv2d(x_data, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x_data):
    return tf.nn.max_pool(x_data, strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME') #ksize means 2x2
def weights_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)
def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#convolutional layer1 + max pooling;
W_cov1 = weights_variable([5,5,1,32])    #means 5x5 patch size, 1 channel input ,32 output channels
b_cov1 = biases_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_cov1 = tf.nn.relu(cov2d(x_image, W_cov1) + b_cov1)
h_pool1 = max_pool_2x2(h_cov1)
#convolutional layer2 + max pooling;
W_cov2 = weights_variable([5,5,32,64])    #h_pool1 is [-1,14,14,32]
b_cov2 = biases_variable([64])
h_cov2 = tf.nn.relu(cov2d(h_pool1, W_cov2) + b_cov2)
h_pool2 = max_pool_2x2(h_cov2)          #h_pool2 is [-1,7,7,64]
#fully connected layer1 + dropout;
W_fc1 = weights_variable([7*7*64,1024])
b_fc1 = biases_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #h_fc1 is [-1,1024]
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#fully connected layer2 to prediction.
W_fc2 = weights_variable([1024, 10])
b_fc2 = biases_variable([10])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2    #h_fc2 is [-1,10]
#train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels= y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0] , y: batch[1], keep_prob: 0.5})
    if i%100 ==0:
        print('step %d,train_accuracy %.4f'%(i, compute_accuracy(batch[0], batch[1])))
    plt.scatter(i,compute_accuracy(batch[0],batch[1]))
plt.ioff()
plt.show()
print('test_accuracy:')
print(compute_accuracy( mnist.test.images, mnist.test.labels))




