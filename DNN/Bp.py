import tensorflow as tf
import numpy as np
from data import generate_data
from data import get_result

#x_train,y_train,x_test,y_test,test_keys,test_truthtable = generate_data.generate_train()
x_train,y_train,x_test,y_test,test_keys,test_truthtable = generate_data.generate_train_saved("test.mashup.csv")
x_test,y_test,test_keys,test_truthtable = generate_data.generate_test_saved("test.mashup.csv")

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

# 1.训练的数据
# Make up some real data 

# 2.定义节点准备接收数据
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 600])
ys = tf.placeholder(tf.float32, [None, 2])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
l1 = add_layer(xs, 600, 200, activation_function=tf.nn.relu)

l2 = add_layer(l1, 200, 100, activation_function=tf.nn.relu)

l3 = add_layer(l2, 100, 20, activation_function=tf.nn.relu)

# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l3, 20, 2, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                    reduction_indices=[1]))
cross_entropy = tf.losses.softmax_cross_entropy(ys, prediction)
# 5.选择 optimizer 使 loss 达到最小                   
# 这一行定义了用什么方式去减少 loss，学习率是 0.1       
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

def compute_accuracy(v_xs, v_ys,test_keys,test_truthtable):
    global prediction
    pred = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(v_ys, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    continue_ = get_result.general(pred,test_keys,test_truthtable)
    return continue_

# 迭代 1000 次学习，sess.run optimizer
for i in range(300):
    #batch_size = len(x_test)
    batch_size = 50000
    data_size = len(x_train)
    start = i * batch_size % data_size
    end = min(start + batch_size,data_size)
    batch_xs, batch_ys = x_train[start:end],y_train[start:end]
   # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print("train ",i)
       # to see the step improvement
       #print(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
       #print(sess.run(prediction,feed_dict = {xs:x_test}))
        continue_ = compute_accuracy(x_test, y_test,test_keys,test_truthtable)
        
        if not continue_:
            break
        