# Combining Everything Together
#----------------------------------
# This file will perform binary classification on the
# iris dataset. We will only predict if a flower is
# I.setosa or not.
#
# We will create a simple binary classifier by creating a line
# and running everything through a sigmoid to get a binary predictor.
# The two features we will use are pedal length and pedal width.
#
# We will use batch training, but this can be easily
# adapted to stochastic training.

import os
import re
import io
import sys
#import requests
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Define App Flags
tf.flags.DEFINE_string("storage_folder", "datas", "Where to store model and data.")
#tf.flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
tf.flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')
tf.flags.DEFINE_integer('batch_size', 200, 'Batch Size for training.')
tf.flags.DEFINE_integer('array_size', 32, 'Array Size for training.')
tf.flags.DEFINE_integer('rnn_size', 15, 'RNN feature size.')
tf.flags.DEFINE_integer('train_times', 10000, 'Train times.')
tf.flags.DEFINE_integer('text_size', 10, 'Text buff size.')

FLAGS = tf.flags.FLAGS

# 网络参数
n_hidden_1 = 32  					#神经网络第一层节点数
n_hidden_2 = 32   					#神经网络第二层节点数
n_input = 2*FLAGS.array_size      	#输入的维度
n_classes = 1      					#输出的维度（分类问题为分类的个数）


# Define how to get data
def get_train_data(storage_folder=FLAGS.storage_folder, data_file="train_data.log"):
	"""
	This function gets the spam/ham data.  It will download it if it doesn't
	already exist on disk (at specified folder/file location).
	"""
	# Make a storage folder for models and data
	if not os.path.exists(storage_folder):
		os.makedirs(storage_folder)

	if not os.path.isfile(os.path.join(storage_folder, data_file)):
		print(data_file+" is not exist!")
	else:
		# Open data from text file
		text_label = []

		text_xa = []
		text_xb = []
		
		print("------get %s data-----"%data_file)

		sessflag_xa = False
		sessId = 0
		Lnum = 0
		Rnum = 0
		with open(os.path.join(storage_folder, data_file), 'r') as file_conn:
			for row in file_conn:
			
				int_list = [i for i in row.split()]
				#print("list:")
				#print(int_list)

				rr = np.array(row[:-1])
				
				if '[' in row and ']' in row:
				
					lableId = row.index('[')+1
					ss = row.index(":")+1
					sessId = row[ss:-1]
					sessId = int(sessId)
					sessflag_xa = False
					#sessflag_lebel = True
					text_label.append(row[lableId])
					if row[lableId] == 'L':
						Lnum = Lnum+1
					elif row[lableId] == 'R':
						Rnum = Rnum+1
					#print("session num:%d"%sessId+"")
				elif sessflag_xa == False:
					
					sessflag_xa = True
					text_xa.append(int_list)
				else:
					
					text_xb.append(int_list)
				#print row info
				#print(rr)

	[lebel, xa_data, xb_data] = [text_label, text_xa, text_xb]
	print("L:%d"%Lnum+"   R:%d"%Rnum)
	return lebel, xa_data, xb_data


# Define accuracy function
def get_accuracy(logits, actuals):
    # Calulate if each output is correct
    batch_acc = tf.equal(tf.argmax(logits, 1), tf.cast(actuals, tf.int64))
    # Convert logical to float
    batch_acc = tf.cast(batch_acc, tf.float32)
    return batch_acc


def accuracy_calc(y_pre, y_tar):
	
	ret = 0

	if y_tar == 1:
		#print("y_tar==1")
		if y_pre[0] > 0:
			ret = 1
	if y_tar == -1:
		#print("y_tar==-1")
		if y_pre[0] < 0:
			ret = 1
	
	print("tar&pre[%d:%d] acc:%d"%(y_tar, y_pre[0],ret))

	return ret



# 创建模型
def multilayer_perceptron(x,weights,biases):

	'''
	# 第一层
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_1 = tf.nn.sigmoid(layer_1)
    
	#第二层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.sigmoid(layer_2)
    
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer
	'''
	out_layer = tf.add(tf.matmul(x, weights['out']), biases['out'])
	
	return out_layer
	
	
#权重的初始化
weights ={
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_input, n_classes]))
}

#偏置的初始化
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Define main program
def main(args):

	Ylebel, Xa, Xb = get_train_data(data_file="train_data.log")

	# Declare batch size
	batch_size = FLAGS.batch_size

	# Create graph
	sess = tf.compat.v1.Session()

	# Declare placeholders
	tf_x_data = tf.compat.v1.placeholder(shape=[None, n_input], dtype=tf.float32)
	y_target  = tf.compat.v1.placeholder(shape=[n_classes, None], dtype=tf.float32)

	# Add model to graph:
	model_output = multilayer_perceptron(tf_x_data, weights, biases)

	# Add classification loss (cross entropy)
	#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
	#xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target)

	# Declare loss function (L2 loss)
	#loss = tf.reduce_mean(tf.square(y_target - model_output))
	#loss = tf.matmul(model_output,-y_target)
	#loss = tf.reduce_mean(tf.square(model_output-y_target))

	#loss = tf.reduce_mean(loss_MinePerceptron(logits=model_output, labels=y_target))
	loss = tf.reduce_mean(tf.where(tf.greater(tf.matmul(y_target, model_output),0), [[0.0]], tf.matmul(-y_target, model_output)))
	#loss = tf.reduce_mean(100.0)
	#loss = tf.reduce_mean(tf.matmul(y_target, model_output))
	#loss = tf.reduce_mean(tf.where(tf.greater(tf.bitwise.bitwise_xor(y_target, model_output),0.0), tf.matmul(-y_target, model_output), [[0.0]]))
	#tf.bitwise_xor
	
	#testing
	#loss = tf.reduce_mean(tf.matmul(model_output,-y_target))#perceptron losser
    
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

	# Create Optimizer
	my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.02)
	#train_step = my_opt.minimize(xentropy)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.compat.v1.global_variables_initializer()
	saver = tf.train.Saver()
	sess.run(init)

	# Run Loop
	for i in range(FLAGS.train_times):
		
		#now set batch size as 1 to reduce difficulty
		rand_index = np.random.choice(len(Ylebel), size=batch_size)
		#print("\r\nrand index:",rand_index)
		
		rand_x_all = []
		rand_y_all = []
		rand_x = []
		rand_y = []
		for idx in rand_index:
			'''
			print(Xa[idx])
			print(Xb[idx])
			'''
			rand_xab = np.array(Xa[idx]+Xb[idx])
			rand_yc = Ylebel[idx]
			'''
			print("~~~~~~~~~~~~~~~~~")
			print(rand_xab)
			print(rand_yc)
			print("~~~~~~~~~~~~~~~~~")
			'''
			#rand_x = np.array([int(x) for x in rand_xab])
			#rand_y = np.array([1 if y=='R' else 0 for y in rand_yc])
			rand_x = [int(x) for x in rand_xab]
			rand_y = [1 if y=='R' else -1 for y in rand_yc]
			
			rand_x_all.append(rand_x)
			rand_y_all.append(rand_y)
			'''
			print("\r\n--------rand_x_all---------")
			print(rand_x_all)
			print("--------rand_y_all---------")
			print(rand_y_all)
			#rand_xT = rand_x.reshape(2*FLAGS.array_size,1)
			#rand_yT = rand_y.reshape(1,1)
			'''
		'''
		rand_xT = rand_x.reshape(2*FLAGS.array_size,1)
		rand_yT = rand_y.reshape(1,1)
		'''

		#print(len(rand_x_all[0]))
		#print(len(rand_y_all[0]))
		
		#translate matrix_T
		#rand_xT = [[row[i] for row in rand_x_all] for i in range(len(rand_x_all[0]))]
		rand_yT = [[row[i] for row in rand_y_all] for i in range(len(rand_y_all[0]))]

		#print('RUN---batch_size %d----'%batch_size)
		#print(rand_xT)
		#print(rand_yT)
		loss_val,output = sess.run([loss,model_output], feed_dict={tf_x_data: rand_x_all, y_target: rand_yT})
		if (i+1)%10==0:
			#print('Step #' + str(i+1) + ' weights = ' + str(sess.run(weights['out'])) + ', b = ' + str(sess.run(biases['out'])))
			print("the train step is %d,loss is %f" % (i+1, loss_val))
		'''
		sess.run(loss, feed_dict={tf_x_data: rand_xT, y_target: rand_yT})
		if (i+1)%10==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
		'''
	
	# Run test loop
	Ylebel_test, Xa_test, Xb_test = get_train_data(data_file="test_data.log")
	print("####################################")
	print(len(Ylebel_test))
	#print(Ylebel_test)
	#print(Xa_test)
	#print(Xb_test)
	
	'''
	# Initialize variables
	init = tf.compat.v1.global_variables_initializer()
	saver = tf.train.Saver()
	sess.run(init)
	'''
	accuracy = 0
	total = 0
	for idx in range(len(Ylebel_test)):
		total = total + 1
		rand_xab_test = np.array(Xa_test[idx]+Xb_test[idx])
		rand_yc_test = Ylebel_test[idx]
		
		rand_y_test = np.array([1 if y=='R' else -1 for y in rand_yc_test])
		rand_x_test_int = np.array([int(x) for x in rand_xab_test])
		rand_x_test = rand_x_test_int.astype(np.float32)
		
		rand_xT_test = rand_x_test.reshape(1, 2*FLAGS.array_size)
		rand_yT_test = rand_y_test.reshape(1, 1)
		
		'''
		print("\r\n--------rand_xT_test---------")
		print(rand_xT_test)
		print("--------rand_yT_test---------")
		print(rand_yT_test)		
		'''
		#TAB
		y_prediction = tf.ones(shape=[1,1], dtype=tf.float32,name='y_prediction')
		# Evaluate Predictions on test set
		
		y_prediction = multilayer_perceptron(tf_x_data, weights, biases)
		
		#correct_prediction = accuracy_calc(y_prediction, y_target)
		#y_prediction = perceptron_model(rand_xT_test, A, b)
		#correct_prediction = accuracy_calc(y_prediction, rand_yT_test)
		
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		y_preddd = sess.run([y_prediction], feed_dict={tf_x_data: rand_xT_test})
		
		print('')
		correct_prediction = accuracy_calc(y_preddd, rand_yT_test)
		accuracy = accuracy + correct_prediction

		acc= tf.zeros(shape=[1,1], dtype=tf.float32,name='acc')
		acc = accuracy/total
		#sess.run(y_prediction)
		#print(y_prediction)
		print('Accuracy: %f'%acc)
		#print(acc)
		#print(x.eval())
		#print(y.eval()) #一次只能打印一个		



# Run main module/tf App
if __name__ == "__main__":
    cmd_args = sys.argv
    if len(cmd_args) > 1 and cmd_args[1] == 'test':
        # Perform unit tests
        tf.test.main(argv=cmd_args[1:])
    else:
        # Run TF App
        tf.compat.v1.app.run(main=None, argv=cmd_args)
		


