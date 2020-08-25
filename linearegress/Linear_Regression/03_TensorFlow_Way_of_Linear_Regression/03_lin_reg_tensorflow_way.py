# Linear Regression: TensorFlow Way
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.compat.v1.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
'''
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
'''
'''
#x1_vals,x2_vals = np.loadtxt('../mydata/1.txt', skiprows=0, unpack=True)
vals = np.loadtxt('1.txt', skiprows=0)
#print(vals)
y1_vals = vals[0]
y2_vals = vals[1]
print("#####################################")
print(y1_vals)
print(y2_vals)
'''
# Declare batch size
batch_size = 5

# Initialize placeholders
x_data = tf.placeholder(shape=[30,batch_size], dtype=tf.float32)
y_target = tf.placeholder(shape=[batch_size,1], dtype=tf.float32)

# Create variables for linear regression
W = tf.Variable(tf.random_normal(shape=[1,30]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
model_output = tf.add(tf.matmul(W, x_data), b)
#model_output = tf.add(np.dot(W, x_data), b)

# Declare loss function (L2 loss)
#loss = tf.reduce_mean(tf.square(y_target - model_output))
loss = tf.matmul(model_output,-y_target)
#loss = tf.reduce_mean(tf.square(model_output-y_target))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
x1_vals = []

for i in range(2):
    #rand_index = np.random.choice(len(x_vals), size=batch_size)
    #rand_x = np.transpose(np.array(x_vals)[rand_index])
    #rand_y = np.transpose(np.array(y1_vals)[rand_index])

    x1_vals.clear()

    for trav in range(batch_size):
        filename = "%d.txt"%(trav+1)
        print(filename)
        vals = np.loadtxt(filename, skiprows=0)
        x1_vals.append(vals[0])   #30 dimensionality
        #x2_vals = vals[1]

    rand_x_reshap = np.array(x1_vals).reshape(batch_size, 30).T
    #rand_y_reshap = np.array(y1_vals).reshape(30, 1)
    print("##################rand_x_reshap###################")
    print(rand_x_reshap)

    rand_y = np.ndarray(shape = (batch_size, 1))
    rand_y = [[1],[1],[1],[0],[0]]
    #rand_y = [-1]

    sess.run(train_step, feed_dict={x_data: rand_x_reshap, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x_reshap, y_target: rand_y})
    loss_vec.append(temp_loss)
    #if (i+1)%6==0:
    print('Step #' + str(i+1) + ' W = ' + str(sess.run(W)) + ' b = ' + str(sess.run(b)))
    print('Loss = ' + str(temp_loss))

'''
# Get the optimal coefficients
#[slope] = sess.run(W)
[slope] = sess.run(W)
[y_intercept] = sess.run(b)

print(slope)
print(y_intercept)

# Get best fit line
best_fit = []
print("best_fit-----")
for i in x_vals:
    print(i)
    print(slope*i+y_intercept)
    best_fit.append(slope*i+y_intercept)
'''
'''
# Plot the result
plt.plot(x_vals, y1_vals, 'o', label='Data Points')
#plt.plot(x_vals, y2_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()
'''
