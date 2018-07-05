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
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#训练模型
#成本函数：交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.initialize_all_variables()

#启动图（graph）
sess = tf.Session()
sess.run(init)

#评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

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
		batch_accuracy = sess.run(accuracy, feed_dict={x:batch[0],y_:batch[1]})
		batch_loss = sess.run(cross_entropy, feed_dict={x:batch[0],y_:batch[1]})
		print("step %d, training accuracy %.5f, loss %.5f"%(i, batch_accuracy, batch_loss))
		train_accuracy = np.append(train_accuracy, batch_accuracy)
		train_loss = np.append(train_loss, batch_loss)
		saver.save(sess, 'softmax_model/mnist-model', global_step=i)
	sess.run(train_step, feed_dict = {x:batch[0],y_:batch[1]})
finish = time.time()
elapsed_time = finish - start

total_accuracy = sess.run(accuracy, feed_dict = {x:mnist.test.images,y_:mnist.test.labels})
print("test accuracy %.5f, elapsed-time %.5f"%(total_accuracy,elapsed_time))

eval_fig = plt.figure()
plt.suptitle('minist softmax regression tarin')
loss_ax = eval_fig.add_subplot(2,1,1)
accu_ax = eval_fig.add_subplot(2,1,2)
loss_ax.set_title('train_loss     elapsed time: %.5f'%elapsed_time, fontsize='x-small')
accu_ax.set_title('train_accuracy', fontsize='x-small')
loss_ax.plot(train_loss, 'r--')
loss_ax.annotate('loss=%.5f'%train_loss[len(train_loss)-1], xy=(iteration/100, train_loss[len(train_loss)-1]), xytext=(-90,20), textcoords='offset points')
accu_ax.plot(train_accuracy, 'b--')
accu_ax.annotate('accuracy=%.5f'%total_accuracy, xy=(iteration/100, train_accuracy[len(train_accuracy)-1]), xytext=(-90,-20), textcoords='offset points')

plt.savefig('softmax-regression-train.png', dpi=400, bbox_inches='tight')
plt.show()
