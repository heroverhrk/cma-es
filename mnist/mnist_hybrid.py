import tensorflow as tf
import cupy as cp
import numpy as np
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

rng = np.random.RandomState(1234)
random_state = 42

epoches = 1000
population = 1000
quantile = 0.6

#batch_size_ev = input('batch>>> ')
#batch_size_ev = int(batch_size_ev)
batch_size_ev = 256

sess = tf.Session()

from datetime import datetime
date = datetime.now().strftime('%x').replace('/', '_')
#filename = 'log/mnist_adam_hyb_' + date + '.txt'
filename = 'log/mnist_sgd_0.01_hyb_noise0.15_' + date + '.txt'
file = open(filename, 'w')

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_x, mnist_y = mnist.train.images, mnist.train.labels
#mnist_x = mnist_x.astype('float64')
train_x, valid_x, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.1)#, random_state=random_state)

in_dim = train_x.shape[1]
out_dim = train_y.shape[1]

def w_quantile(population, quantile):
    return np.append(np.zeros(int(population*quantile)).astype('float32'), np.ones(int(population - population*quantile)).astype('float32'))

def relu_(x):
    return cp.fmax(x, 0)

class dense:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_mean = np.zeros(in_dim*out_dim).astype('float64')
        self.W_cov = np.ones(in_dim*out_dim).astype('float64') * 5.0
        self.b_mean = np.zeros(out_dim).astype('float64')
        self.b_cov = np.ones(out_dim).astype('float64') * 5.0
        self.W_p_sig = 0
        self.W_p_c = 0
        self.W_sigma = np.ones(in_dim*out_dim)
        self.b_p_sig = 0
        self.b_p_c = 0
        self.b_sigma = np.ones(out_dim)
        self.W_c_sig = 3/(in_dim*out_dim)
        self.b_c_sig = 3/out_dim

    def gen_ind(self, population):
        self.population = population
        W_mean = cp.asnumpy(self.W_mean)
        W_cov = cp.asnumpy(self.W_cov)
        b_mean = cp.asnumpy(self.b_mean)
        b_cov = cp.asnumpy(self.b_cov)
        mnd_W = tf.contrib.distributions.MultivariateNormalDiag(W_mean, W_cov)
        mnd_b = tf.contrib.distributions.MultivariateNormalDiag(b_mean, b_cov)
        self.W = mnd_W.sample(population)
        self.b = mnd_b.sample(population)
        self.cost = np.zeros(population).astype('float32')
        self.f1 = np.zeros(population).astype('float32')

    def conv_to_cp(self):
        self.W = sess.run(self.W)
        self.W = cp.array(self.W.astype('float32')).reshape(self.population, self.in_dim, self.out_dim)
        self.b = sess.run(self.b)
        self.b = cp.array(self.b.astype('float32'))
        self.cost = cp.array(self.cost)
        self.f1 = cp.array(self.f1)

    def selection(self):
        index = cp.argsort(-self.cost)
        index = cp.asnumpy(index)
        self.W = cp.asnumpy(self.W)
        self.b = cp.asnumpy(self.b)
        self.W = self.W[index]
        self.b = self.b[index]
        #self.W = cp.array(self.W)
        #self.b = cp.array(self.b)
        self.W_top = cp.asnumpy(self.W[self.population-1])
        self.b_top = cp.asnumpy(self.b[self.population-1])

    def update_path(self, quantile):
        w = 1/self.population * w_quantile(self.population, quantile)
        w /= np.sum(w)
        w = w[:, np.newaxis]

        eta_m = 1.0
        eta_c = 1.0 / ((self.in_dim*self.out_dim)**2 * np.sum(w**2))

        #self.W_mean = cp.array(self.W_mean)
        #self.W_cov = cp.array(self.W_cov)
        #self.b_mean = cp.array(self.b_mean)
        #self.b_cov = cp.array(self.b_cov)

        W_mean_ = self.W_mean
        b_mean_ = self.b_mean

        self.W_mean = self.W_mean + eta_m * np.sum(w * (np.reshape(self.W, [self.population, self.in_dim*self.out_dim]) - self.W_mean), axis=0)
        self.W_p_sig = (1-self.W_c_sig)*self.W_p_sig + np.sqrt(1-(1-self.W_c_sig)**2)*np.sqrt(1/np.sum(w**2))*np.sqrt(1/self.W_cov)*(self.W_mean-W_mean_)/self.W_sigma
        self.W_sigma = self.W_sigma * np.exp(self.W_c_sig*(np.linalg.norm(self.W_p_sig)/(math.sqrt(self.in_dim*self.out_dim)*(1-1/(4*self.in_dim*self.out_dim)+1/(21*((self.in_dim*self.out_dim)**2))))-1))
        self.b_mean = self.b_mean + eta_m * np.sum(w * (self.b - self.b_mean), axis=0)
        self.b_p_sig = (1-self.b_c_sig)*self.b_p_sig + np.sqrt(1-(1-self.b_c_sig)**2)*np.sqrt(1/np.sum(w**2))*np.sqrt(1/self.b_cov)*(self.b_mean-b_mean_)/self.b_sigma
        self.b_sigma = self.b_sigma * np.exp(self.b_c_sig*(np.linalg.norm(self.b_p_sig)/(math.sqrt(self.out_dim)*(1-1/(4*self.out_dim)+1/(21*(self.out_dim**2))))-1))

        self.W_cov = self.W_cov + eta_c * np.sum(w * (((np.reshape(self.W, [self.population, self.in_dim*self.out_dim]) - W_mean_)/self.W_sigma)**2 - self.W_cov), axis=0)
        self.b_cov = self.b_cov + eta_c * np.sum(w * (((self.b - b_mean_)/self.b_sigma)**2 - self.b_cov), axis=0)

def label_noise(label, rate):
    label_ = np.argmax(label, 1)
    length = label.shape[0]
    index = int(length*rate)
    for i in range(index):
        label_[i] = int(np.random.rand())*10
    label_ = np.eye(10)[label_]
    print(label_.shape)
    return label_

n_batches_ev = train_x.shape[0] // batch_size_ev

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

aa = 6/400

W_tf_1 = tf.Variable(rng.uniform(low=-np.sqrt(6/984), high=np.sqrt(6/984), size=(784,200)).astype('float32'))
b_tf_1 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_2 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_2 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_3 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_3 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_4 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_4 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_5 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_5 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_6 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_6 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_7 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_7 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_8 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_8 = tf.Variable(np.zeros(200).astype('float32'))
W_tf_9 = tf.Variable(rng.uniform(low=-np.sqrt(aa), high=np.sqrt(aa), size=(200,200)).astype('float32'))
b_tf_9 = tf.Variable(np.zeros(200).astype('float32'))

W_tf_10 = tf.Variable(rng.uniform(low=-np.sqrt(6/210), high=np.sqrt(6/210), size=(200,10)).astype('float32'))
b_tf_10 = tf.Variable(np.zeros(10).astype('float32'))

u1 = tf.matmul(x, W_tf_1) + b_tf_1
z1 = tf.nn.relu(u1)
#z1 = tf.nn.dropout(z1, keep_prob)
u2 = tf.matmul(z1, W_tf_2) + b_tf_2
z2 = tf.nn.relu(u2)
u3 = tf.matmul(z2, W_tf_3) + b_tf_3
z3 = tf.nn.relu(u3)
u4 = tf.matmul(z3, W_tf_4) + b_tf_4
z4 = tf.nn.relu(u4)
u5 = tf.matmul(z4, W_tf_5) + b_tf_5
z5 = tf.nn.relu(u5)
u6 = tf.matmul(z5, W_tf_6) + b_tf_6
z6 = tf.nn.relu(u6)
u7 = tf.matmul(z6, W_tf_7) + b_tf_7
z7 = tf.nn.relu(u7)
u8 = tf.matmul(z7, W_tf_8) + b_tf_8
z8 = tf.nn.relu(u8)
u9 = tf.matmul(z8, W_tf_9) + b_tf_9
z9 = tf.nn.relu(u9)
u10 = tf.matmul(z9, W_tf_10) + b_tf_10
y = tf.nn.softmax(u10)
cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#train = tf.train.AdamOptimizer(0.001).minimize(cost)
valid = tf.argmax(y, 1)

epoches_sgd = 20

batch_size = 64
n_batches = train_x.shape[0] // batch_size
f1_best = 0.0

init = tf.global_variables_initializer()
#init1 = tf.variables_initializer([W_tf_2, b_tf_2, W_tf_3, b_tf_3, W_tf_4, b_tf_4, W_tf_1, b_tf_1])
#sess.run(init1)
sess.run(init)

dense1 = dense(784, 200)

print('--Start Generation--')
train_x, train_y = shuffle(train_x, train_y, random_state=random_state)
train_y = label_noise(train_y, 0.15)

for epoch in range(epoches):
    train_x, train_y = shuffle(train_x, train_y, random_state=random_state)

    for gen in range(n_batches_ev):

        W2 = cp.array(sess.run(W_tf_2))
        b2 = cp.array(sess.run(b_tf_2))
        W3 = cp.array(sess.run(W_tf_3))
        b3 = cp.array(sess.run(b_tf_3))
        W4 = cp.array(sess.run(W_tf_4))
        b4 = cp.array(sess.run(b_tf_4))
        W5 = cp.array(sess.run(W_tf_5))
        b5 = cp.array(sess.run(b_tf_5))
        W6 = cp.array(sess.run(W_tf_6))
        b6 = cp.array(sess.run(b_tf_6))
        W7 = cp.array(sess.run(W_tf_7))
        b7 = cp.array(sess.run(b_tf_7))
        W8 = cp.array(sess.run(W_tf_8))
        b8 = cp.array(sess.run(b_tf_8))
        W9 = cp.array(sess.run(W_tf_9))
        b9 = cp.array(sess.run(b_tf_9))
        W10 = cp.array(sess.run(W_tf_10))
        b10 = cp.array(sess.run(b_tf_10))

        cp.cuda.Device(0).use()
        dense1.gen_ind(population)
        dense1.conv_to_cp()

        start = gen * batch_size_ev
        end = start + batch_size_ev


        for i in range(start, end):
            train_x_cp = cp.array(train_x[i])
            train_y_cp = cp.array(train_y[i])
            y = relu_(cp.matmul(train_x_cp, dense1.W) + dense1.b)
            y = relu_(cp.matmul(y, W2) + b2)
            y = relu_(cp.matmul(y, W3) + b3)
            y = relu_(cp.matmul(y, W4) + b4)
            y = relu_(cp.matmul(y, W5) + b5)
            y = relu_(cp.matmul(y, W6) + b6)
            y = relu_(cp.matmul(y, W7) + b7)
            y = relu_(cp.matmul(y, W8) + b8)
            y = relu_(cp.matmul(y, W9) + b9)
            y = cp.matmul(y, W10) + b10
            y = cp.exp(y) / cp.sum(cp.exp(y), axis=1)[:, cp.newaxis]
            y = cp.fmax(1e-10, y)
            dense1.cost += -cp.sum(train_y_cp*cp.log(y), axis=1)
        dense1.cost /= batch_size_ev
        dense1.selection()
        dense1.update_path(quantile)

        W1 = cp.array(dense1.W_top)
        b1 = cp.array(dense1.b_top)

        valid_x_cp = cp.array(valid_x)
        valid_y_cp = cp.array(valid_y)
        valid_y = cp.asnumpy(valid_y)
        y = relu_(cp.matmul(valid_x_cp, W1) + b1)
        y = relu_(cp.matmul(y, W2) + b2)
        y = relu_(cp.matmul(y, W3) + b3)
        y = relu_(cp.matmul(y, W4) + b4)
        y = relu_(cp.matmul(y, W5) + b5)
        y = relu_(cp.matmul(y, W6) + b6)
        y = relu_(cp.matmul(y, W7) + b7)
        y = relu_(cp.matmul(y, W8) + b8)
        y = relu_(cp.matmul(y, W9) + b9)
        y = cp.matmul(y, W10) + b10
        y = cp.exp(y) / (cp.sum(cp.exp(y), axis=1)[:, cp.newaxis])
        #y = cp.clip(cp.exp(cp.matmul(valid_x[i], network.W) + network.b) / cp.sum(cp.exp(cp.matmul(valid_x[i], network.W) + network.b)), 1e-10, 1.0)
        y = cp.fmax(1e-10, y)
        valid_cost = cp.mean(-cp.sum(valid_y_cp*cp.log(y), axis=1))
        #valid_cost = cp.fmax(0, valid_cost)
        y = cp.asnumpy(y)
        pred_y = np.argmax(y, 1).astype('int32')
        f1 = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'), average='macro')
        if(f1 > f1_best):
            f1_best = f1
        acc = accuracy_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'))
        print('EP:%i, GEN:%i (%i / %i), Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, gen+1, (gen+1)*batch_size_ev, train_x.shape[0], valid_cost, f1, acc, f1_best))
        print('EP:%i, GEN:%i (%i / %i), Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, gen+1, (gen+1)*batch_size_ev, train_x.shape[0], valid_cost, f1, acc, f1_best), file=file)

        #x = tf.placeholder(tf.float32, [None, 784])
        #t = tf.placeholder(tf.float32, [None, 10])

        W_tf_1 = tf.Variable(cp.asnumpy(W1))
        b_tf_1 = tf.Variable(cp.asnumpy(b1))

        u1 = tf.matmul(x, W_tf_1) + b_tf_1
        z1 = tf.nn.relu(u1)
        #z1 = tf.nn.dropout(z1, keep_prob)
        u2 = tf.matmul(z1, W_tf_2) + b_tf_2
        z2 = tf.nn.relu(u2)
        u3 = tf.matmul(z2, W_tf_3) + b_tf_3
        z3 = tf.nn.relu(u3)
        u4 = tf.matmul(z3, W_tf_4) + b_tf_4
        z4 = tf.nn.relu(u4)
        u5 = tf.matmul(z4, W_tf_5) + b_tf_5
        z5 = tf.nn.relu(u5)
        u6 = tf.matmul(z5, W_tf_6) + b_tf_6
        z6 = tf.nn.relu(u6)
        u7 = tf.matmul(z6, W_tf_7) + b_tf_7
        z7 = tf.nn.relu(u7)
        u8 = tf.matmul(z7, W_tf_8) + b_tf_8
        z8 = tf.nn.relu(u8)
        u9 = tf.matmul(z8, W_tf_9) + b_tf_9
        z9 = tf.nn.relu(u9)
        u10 = tf.matmul(z9, W_tf_10) + b_tf_10
        y = tf.nn.softmax(u10)
        cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
        train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        #train = tf.train.AdamOptimizer(0.001).minimize(cost)
        valid = tf.argmax(y, 1)
        #init = tf.global_variables_initializer()
        init0 = tf.variables_initializer([W_tf_1, b_tf_1])
        sess.run(init0)
        #sess.run(init)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: train_x[start:end], t: train_y[start:end], keep_prob: 1.0})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_x, t: valid_y, keep_prob: 1.0})
            #valid_cost /= valid_X.shape[0]
        f1 = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'), average='macro')
        if(f1 > f1_best):
            f1_best = f1
        acc = accuracy_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'))
        print('EP:%i, Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, valid_cost, f1, acc, f1_best))
        print('EP:%i, Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, valid_cost, f1, acc, f1_best), file=file)
file.close()
