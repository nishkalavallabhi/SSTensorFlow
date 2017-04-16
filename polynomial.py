#Polynomial regression with one variable, degree 6

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 40

trX = np.linspace(-1, 1, 101)
num_coeffs = 6

def functionY(trX):
 trY_coeffs = [1, 2, 3, 4, 5, 6]
 trY = 0
 for i in range(num_coeffs):
   trY += trY_coeffs[i] * np.power(trX, i)
 trY += np.random.randn(*trX.shape) * 1.5
 return trY

trY = functionY(trX)

plt.scatter(trX, trY)
plt.savefig("polyscatter.png")

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X, w): 
  terms = []
  for i in range(num_coeffs): 
     term = tf.multiply(w[i], tf.pow(X, i))
     terms.append(term)
  return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)
cost = tf.square(Y-y_model)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
   for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})
w_val = sess.run(w)
print("Learned weights", w_val)
sess.close()


trYpredict = 0

testX = np.linspace(-1, 1, 51)
print("Generated test data", testX)
testY = functionY(testX)
testYpredict = 0

for i in range(num_coeffs):
  testYpredict += w_val[i] * np.power(testX, i)

#Evaluate the model with test data
avg_error = 0
for (y,yp) in zip(testY,testYpredict):
  avg_error += abs(y-yp)

print("avg error on test data", avg_error/len(testY))
