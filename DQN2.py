import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class DQN2:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet2'
        self.sess = tf.compat.v1.Session()
        self.x = tf.compat.v1.placeholder('float', [None, params['width'], params['height'], 6], name=self.network_name + '_x')
        self.q_t = tf.compat.v1.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.compat.v1.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_terminals')

        # Layer 1 (Convolutional)
        layer_name = 'conv1'; size = 3; channels = 6; filters = 32; stride = 1
        self.w1 = tf.Variable(tf.random.normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c1 = tf.nn.conv2d(self.x, filters=self.w1, strides=[1, stride, stride, 1], padding='SAME', name=self.network_name + '_' + layer_name + '_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name + '_' + layer_name + '_activations')

        o1_shape = self.o1.get_shape().as_list()

        # Layer 2 (Fully connected)
        layer_name = 'fc2'; hiddens = 4; dim = o1_shape[1] * o1_shape[2] * o1_shape[3]
        self.o1_flat = tf.reshape(self.o1, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
        self.w2 = tf.Variable(tf.random.normal([dim, hiddens], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
        self.y = tf.add(tf.matmul(self.o1_flat, self.w2), self.b2, name=self.network_name + '_' + layer_name + '_outputs')

        # Q, Cost, Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y, self.actions), axis=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optim = tf.compat.v1.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess, self.params['load_file'])

    def train(self, bat_s, bat_a, bat_t, bat_n, bat_r):
        feed_dict = {self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals: bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y, feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict = {self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals: bat_t, self.rewards: bat_r}
        _, cnt, cost = self.sess.run([self.optim, self.global_step, self.cost], feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self, filename):
        self.saver.save(self.sess, filename)
