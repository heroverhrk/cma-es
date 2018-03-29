import tensorflow as tf
import cupy as cp
import numpy as np
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10

nan_ = 1.0e-10
inf_ = 1.0e10

rng = cp.random.RandomState(1234)
random_state = 42
ngen = 100
population = 4000
quantile = 0.6

#batch_size_ev = input('batch >>> ')
#batch_size_ev = int(batch_size_ev)
batch_size_ev = 256

from datetime import datetime
date = datetime.now().strftime('%x').replace('/', '_')
filename = 'log/cifar10_cupy_noise0.2_' + date + '.txt'
file = open(filename, 'w')

dim = 32 * 32

(cifar_x_1, cifar_y_1), (cifar_x_2, cifar_y_2) = cifar10.load_data()

cifar_x = np.r_[cifar_x_1, cifar_x_2]
cifar_y = np.r_[cifar_y_1, cifar_y_2]

cifar_x = cifar_x.astype('float32') / 255
cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

cifar_x = np.reshape(cifar_x, (60000, 32*32, 3))

cifar_x = 0.299*cifar_x[:,:,0] + 0.587*cifar_x[:,:,1] + 0.114*cifar_x[:,:,2]

train_x, test_x, train_y, test_y = train_test_split(cifar_x, cifar_y, test_size=10000, random_state=random_state)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=10000, random_state=random_state)

#cp.cuda.Device(1).use()
valid_x_cp = cp.array(valid_x)
valid_y_cp = cp.array(valid_y)
#valid_y = cp.asnumpy(valid_y_cp)

class dense:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_mean = np.zeros(in_dim*out_dim).astype('float64')
        self.W_cov = np.ones(in_dim*out_dim).astype('float64')*5
        self.b_mean = np.zeros(out_dim).astype('float64')
        self.b_cov = np.ones(out_dim).astype('float64')*5
        self.W_p_sig = 0
        self.W_p_c = 0
        self.W_sigma = cp.ones(in_dim*out_dim)
        self.b_p_sig = 0
        self.b_p_c = 0
        self.b_sigma = cp.ones(out_dim)
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
        self.W = cp.array(self.W.astype('float64')).reshape(self.population, self.in_dim, self.out_dim)
        self.b = sess.run(self.b)
        self.b = cp.array(self.b.astype('float64'))
        self.cost = cp.array(self.cost)
        self.f1 = cp.array(self.f1)

    def selection(self):
        index = cp.argsort(-self.cost)
        self.W = self.W[index]
        self.b = self.b[index]
        self.W_top = cp.array(self.W[self.population-1])
        self.b_top = cp.array(self.b[self.population-1])

    def update_path(self, quantile):
        w = 1/self.population * w_quantile(self.population, quantile)
        w /= cp.sum(w)
        w = w[:, cp.newaxis]

        eta_m = 1.0
        eta_c = 1.0 / ((self.in_dim*self.out_dim)**2 * cp.sum(w**2))

        self.W_mean = cp.array(self.W_mean)
        self.W_cov = cp.array(self.W_cov)
        self.b_mean = cp.array(self.b_mean)
        self.b_cov = cp.array(self.b_cov)

        W_mean_ = self.W_mean
        b_mean_ = self.b_mean

        self.W_mean = self.W_mean + eta_m * cp.sum(w * (cp.reshape(self.W, [self.population, self.in_dim*self.out_dim]) - self.W_mean), axis=0)
        self.W_p_sig = (1-self.W_c_sig)*self.W_p_sig + cp.sqrt(1-(1-self.W_c_sig)**2)*cp.sqrt(1/np.sum(w**2))*cp.sqrt(1/self.W_cov)*(self.W_mean-W_mean_)/self.W_sigma
        self.W_sigma = self.W_sigma * cp.exp(self.W_c_sig*(cp.linalg.norm(self.W_p_sig)/(math.sqrt(self.in_dim*self.out_dim)*(1-1/(4*self.in_dim*self.out_dim)+1/(21*((self.in_dim*self.out_dim)**2))))-1))
        self.b_mean = self.b_mean + eta_m * cp.sum(w * (self.b - self.b_mean), axis=0)
        self.b_p_sig = (1-self.b_c_sig)*self.b_p_sig + cp.sqrt(1-(1-self.b_c_sig)**2)*cp.sqrt(1/np.sum(w**2))*cp.sqrt(1/self.b_cov)*(self.b_mean-b_mean_)/self.b_sigma
        self.b_sigma = self.b_sigma * cp.exp(self.b_c_sig*(cp.linalg.norm(self.b_p_sig)/(math.sqrt(self.out_dim)*(1-1/(4*self.out_dim)+1/(21*(self.out_dim**2))))-1))

        self.W_cov = self.W_cov + eta_c * cp.sum(w * (((cp.reshape(self.W, [self.population, self.in_dim*self.out_dim]) - W_mean_)/self.W_sigma)**2 - self.W_cov), axis=0)
        self.b_cov = self.b_cov + eta_c * cp.sum(w * (((self.b - b_mean_)/self.b_sigma)**2 - self.b_cov), axis=0)

def sigmoid(x):
    return 1.0 / (1.0 + cp.exp(-x))

def w_quantile(population, quantile):
    w = np.append(np.zeros(int(population*quantile)).astype('float32'), np.ones(int(population - population*quantile)).astype('float32'))
    w = cp.array(w)
    return w

def label_noise(label, rate):
    label_ = np.argmax(label, 1)
    length = label.shape[0]
    index = int(length*rate)
    for i in range(index):
        label_[i] = int(np.random.rand())*10
    label_ = np.eye(10)[label_]
    return label_

#batch_size = 128
n_batches = train_x.shape[0] // batch_size_ev

f1_best = 0.0

sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
network = dense(dim, 10)

print('--Start Generation--')
train_x, train_y = shuffle(train_x, train_y, random_state=random_state)
train_y = label_noise(train_y, 0.2)
for epoch in range(ngen):
    train_x, train_y = shuffle(train_x, train_y, random_state=random_state)
    for gen in range(n_batches):
        cp.cuda.Device(0).use()
        valid_y = cp.array(valid_y)
        network.gen_ind(population)
        network.conv_to_cp()
        start = gen * batch_size_ev
        end = start + batch_size_ev
        for i in range(start, end):
            train_x_cp = cp.array(train_x[i])
            train_y_cp = cp.array(train_y[i])
            y = cp.matmul(train_x_cp, network.W) + network.b
            y = cp.exp(y)
            y = y / (cp.sum(y, axis=1)[:, cp.newaxis])
            y = cp.fmax(nan_, y)
            network.cost += -cp.sum(train_y_cp*cp.log(y), axis=1)
        network.cost /= batch_size_ev
        network.selection()
        network.update_path(quantile)
        valid_y = cp.asnumpy(valid_y)
        #cp.cuda.Device(1).use()

        y = cp.matmul(valid_x_cp, network.W_top) + network.b_top
        y = cp.exp(y) / cp.sum(cp.exp(y), axis=1)[:, cp.newaxis]
        y = cp.fmax(1e-10, y)
        valid_cost = cp.mean(-cp.sum(valid_y_cp*cp.log(y), axis=1))
        y = cp.asnumpy(y)
        pred_y = np.argmax(y, 1).astype('int32')
        f1 = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'), average='macro')
        if(f1 > f1_best):
            f1_best = f1
        acc = accuracy_score(np.argmax(valid_y, 1).astype('int32'), pred_y.astype('int32'))
        print('EP:%i, GEN:%i (%i/%i), Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, gen+1, (gen+1)*batch_size_ev, train_x.shape[0], valid_cost, f1, acc, f1_best))
        print('EP:%i, GEN:%i (%i/%i), Valid Cost: %.3f, Valid F1: %.3f, Valid Acc: %.3f, Best F1: %.3f' % (epoch+1, gen+1, (gen+1)*batch_size_ev, train_x.shape[0], valid_cost, f1, acc, f1_best), file=file)
file.close()
