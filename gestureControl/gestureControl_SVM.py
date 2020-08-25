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
tf.flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
tf.flags.DEFINE_float('dropout_prob', 0.5, 'Per to keep probability for dropout.')
tf.flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
tf.flags.DEFINE_integer('batch_size', 10, 'Batch Size for training.')
tf.flags.DEFINE_integer('array_size', 32, 'Array Size for training.')
tf.flags.DEFINE_integer('rnn_size', 15, 'RNN feature size.')
tf.flags.DEFINE_integer('train_times', 100, 'Train times.')
tf.flags.DEFINE_integer('text_size', 10, 'Text buff size.')

FLAGS = tf.flags.FLAGS


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

# Define Perceptron Model
def perceptron_model(x, A, b):
	# Add model to graph:
	my_mult = tf.matmul(A, x)
	my_output = np.add(my_mult, b)
	return my_output

# Define accuracy function
def get_accuracy(logits, actuals):
    # Calulate if each output is correct
    batch_acc = tf.equal(tf.argmax(logits, 1), tf.cast(actuals, tf.int64))
    # Convert logical to float
    batch_acc = tf.cast(batch_acc, tf.float32)
    return batch_acc
'''
def loss_MinePerceptron(logits, labels):
	loss = tf.reduce_sum(tf.where(tf.equal(logits, labels), 0, -(labels)*logits))
	return loss
'''



def accuracy_calc(y_pre, y_tar):
	
	ret = 0
	print('y_tar:%d'%y_tar)
	print('y_pre:%d'%y_pre[0])

	if y_tar == 1:
		print("y_tar==1")
		if y_pre[0] > 0:
			ret = 1
	if y_tar == -1:
		print("y_tar==-1")
		if y_pre[0] < 0:
			ret = 1
	
	return ret

# Define main program
def main(args):
	# Load the data
	#
	'''
	iris = datasets.load_iris()
	print(iris)
	binary_target = np.array([1. if x==0 else 0. for x in iris.target])
	print("--------------------")
	print(binary_target)
	iris_2d = np.array([[x[2], x[3]] for x in iris.data])
	print("--------------------")
	print(iris_2d)
	'''
	
	#Xa = tf.compat.v1.placeholder(shape=[None, FLAGS.array_size])
	#Xb = tf.compat.v1.placeholder(shape=[None, FLAGS.array_size])
	#Xa = [None][32]
	#Xb = [None][32]
	Ylebel, Xa, Xb = get_train_data(data_file="train_data.log")

	# Declare batch size
	batch_size = FLAGS.batch_size

	# Create graph
	sess = tf.compat.v1.Session()

	# Declare placeholders
	tf_x_data = tf.compat.v1.placeholder(shape=[FLAGS.array_size*2, None], dtype=tf.float32)
	#tf_xb_data = tf.compat.v1.placeholder(shape=[1, FLAGS.array_size], dtype=tf.float32)
	y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables A and b (0 = x1 - A*x2 + b)
	A = tf.Variable(tf.random.normal(shape=[1, 2*FLAGS.array_size]))
	b = tf.Variable(tf.random.normal(shape=[1, 1]))

	# Add model to graph:
	model_output = perceptron_model(tf_x_data, A, b)

	# Add classification loss (cross entropy)
	#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
	#xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target)

	# Declare loss function (L2 loss)
	#loss = tf.reduce_mean(tf.square(y_target - model_output))
	#loss = tf.matmul(model_output,-y_target)
	#loss = tf.reduce_mean(tf.square(model_output-y_target))

	#loss = tf.reduce_mean(loss_MinePerceptron(logits=model_output, labels=y_target))
	loss = tf.reduce_mean(tf.where(tf.greater(tf.matmul(model_output, y_target),0.0), [[0.0]], tf.matmul(model_output,-y_target)))
	
	
	#SVM
	hings = tf.losses.hinge_loss(labels=y, logits=y_pred, weights)
	hings_loss = tf.reduce_mean(hings)

	
	#testing
	#loss = tf.reduce_mean(tf.matmul(model_output,-y_target))#perceptron losser

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
		print("\r\nrand index:",rand_index)
		
		rand_x_all = []
		rand_y_all = []
		rand_x = []
		rand_y = []
		for idx in rand_index:
			print("idx in rand_index")
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
		rand_xT = [[row[i] for row in rand_x_all] for i in range(len(rand_x_all[0]))]
		rand_yT = rand_y_all
		#rand_yT = [[row[i] for row in rand_y_all] for i in range(len(rand_y_all[0]))]
		
		print('RUN---batch_size 10----')
		#print(rand_xT)
		#print(rand_yT)
		loss_val,step = sess.run([loss,train_step], feed_dict={tf_x_data: rand_xT, y_target: rand_yT})
		if (i+1)%10==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
			print("the step is,loss is %f" % loss_val)
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
		
		rand_y_test = np.array([1 if y=='R' else 0 for y in rand_yc_test])
		rand_x_test_int = np.array([int(x) for x in rand_xab_test])
		rand_x_test = rand_x_test_int.astype(np.float32)
		
		rand_xT_test = rand_x_test.reshape(2*FLAGS.array_size,1)
		rand_yT_test = rand_y_test.reshape(1,1)
		'''
		print("\r\n--------rand_xT_test---------")
		print(rand_xT_test)
		print("--------rand_yT_test---------")
		print(rand_yT_test)		
		'''
		#TAB
		y_prediction = tf.ones(shape=[1,1], dtype=tf.float32,name='y_prediction')
		# Evaluate Predictions on test set
		#y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(perceptron_model(tf_x_data, A, b))))
		#print(y_prediction)
		
		y_prediction = perceptron_model(tf_x_data, A, b)
		#correct_prediction = accuracy_calc(y_prediction, y_target)
		#y_prediction = perceptron_model(rand_xT_test, A, b)
		#correct_prediction = accuracy_calc(y_prediction, rand_yT_test)
		
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#x = tf.ones(shape=[2,3], dtype=tf.int32,name='x')
		#y= tf.zeros(shape=[2,3], dtype=tf.float32,name='y')
		

		
		#with tf.Session() as sess:
		#with sess.as_default():
		#print(sess.run([x,y]))   #一次能打印两个
		
		y_preddd = sess.run([y_prediction], feed_dict={tf_x_data: rand_xT_test})
		
		print('')
		correct_prediction = accuracy_calc(y_preddd, rand_yT_test)
		print('return accuracy: %d' % correct_prediction)
		accuracy = accuracy + correct_prediction

		acc= tf.zeros(shape=[1,1], dtype=tf.float32,name='acc')
		acc = accuracy/total
		#sess.run(y_prediction)
		#print(y_prediction)
		print('Accuracy: ')
		print(acc)
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
		


