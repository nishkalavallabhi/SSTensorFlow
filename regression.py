#Linear regression with two variables, using gradient descent.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

learning_rate = 0.01
training_epochs = 100

#np.linspace generates 101 equidistant numbers between -1 and +1
x_train = np.linspace(-1, 1, 101) 
z_train = np.linspace(-2, 2, 101)

#creating another set of variables that are 2*x + some random noise added.
#.shape gives the dimensions of x_train and is passed as a *args variable.
y_train = (2 * x_train) + (3 * z_train) + np.random.randn(*x_train.shape) * 0.33

#Creating three place holder variables for the learning.
X = tf.placeholder("float")
Y = tf.placeholder("float")
Z = tf.placeholder("float")

w1 = tf.Variable(0.0, name="weights")
w2 = tf.Variable(0.0, name="weights")

y_model = tf.add(tf.multiply(X, w1), tf.multiply(Z, w2))
#print(y_model)
cost = tf.square(Y-y_model)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

#print(sess.run(w1))
#print(sess.run(w2))

for epoch in range(training_epochs):
   for (x, y, z) in zip(x_train, y_train, z_train):
        sess.run(train_op, feed_dict={X: x, Y: y, Z:z})
        #print(sess.run(w1), "   ", sess.run(w2))

w1_val = sess.run(w1)
w2_val = sess.run(w2)

sess.close()

print("final value for first feature weight", w1_val)
print("final value for second feature weight", w2_val)

plt.scatter(x_train, y_train)
y_learned = (x_train*w1_val) + (z_train*w2_val)

#Simulate test data:
x_test = np.linspace(-1, 1, 50) 
z_test = np.linspace(-2, 2, 50)
y_test = (2 * x_train) + (3 * z_train) + np.random.randn(*x_train.shape) * 0.33
y_learnedtest = (x_test*w1_val) + (z_test*w2_val)

#Evaluate the model with test data
avg_error = 0
for (y,yp) in zip(y_test,y_learnedtest):
  avg_error += abs(y-yp)

print("avg error on test data", avg_error/len(y_train))

'''
plt.plot(x_train, y_learned, 'y')
plt.savefig("regX.png")

plt.plot(z_train, y_learned, 'y')
plt.savefig("regZ.png")
'''
