#下载引入数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

#设置默认会话（session）
sess = tf.InteractiveSession()

#设置模型变量
x = tf.placeholder(tf.float32, [None, 784])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)

#权重初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#卷积与池化
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#构建模型
#第一层卷积
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练模型
#成本函数：交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#评估模型
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#初始化变量#启动图（graph）
sess.run(tf.initialize_all_variables())

#save models
saver = tf.train.Saver(max_to_keep=4)

#训练
train_accuracy = np.empty([1])
train_loss = np.empty([1])
iteration = 20000
start = time.time()

for i in range(iteration):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		batch_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
		batch_loss = cross_entropy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
		print("step %d, training accuracy %.5f, loss %.5f"%(i, batch_accuracy, batch_loss))
		train_accuracy = np.append(train_accuracy, batch_accuracy)
		train_loss = np.append(train_loss, batch_loss)
		saver.save(sess, 'CNN_model/mnist-model', global_step=i)
	train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})
finish = time.time()
elapsed_time = finish - start

total_accuracy = accuracy.eval(feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
print("test accuracy %.5f"%total_accuracy)

eval_fig = plt.figure()
plt.suptitle('minist CNN network tarin')
loss_ax = eval_fig.add_subplot(2,1,1)
accu_ax = eval_fig.add_subplot(2,1,2)
loss_ax.set_title('train_loss     elapsed time: %.5f'%elapsed_time, fontsize='x-small')
accu_ax.set_title('train_accuracy', fontsize='x-small')
loss_ax.plot(train_loss, 'r--')
loss_ax.annotate('loss=%.5f'%train_loss[len(train_loss)-1], xy=(iteration/100, train_loss[len(train_loss)-1]), xytext=(-90,20), textcoords='offset points')
accu_ax.plot(train_accuracy, 'b--')
accu_ax.annotate('accuracy=%.5f'%total_accuracy, xy=(iteration/100, train_accuracy[len(train_accuracy)-1]), xytext=(-90,-20), textcoords='offset points')

plt.savefig('CNN-train.png', dpi=400, bbox_inches='tight')
plt.show()
