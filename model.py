import tensorflow as tf


class Hyperrhncell(object):
  """A recurrent hightway hypernetwork cell."""

  def __init__(self, h, m, depth, inputs, drop, batch_norm=True):
	
        self._n = inputs.shape[-1]
        self._h, self._m, self._depth, self._drop = h, m, depth, drop, 		# self.h and self.m are hypernetwork and main network dims.

        self._inputs = inputs
        self.batch_norm = batch_norm
        self._initial_hidden = self._inputs[:, 0, :]
        self._initial_hidden = tf.matmul(
            self._initial_hidden, tf.zeros([self._n, self._m]))
        self._w_up = []
        self._b_up = []
        self._w_h = []
        self._b_h = []
        self._w_m = []
        self._b_m = []

        for l in range(self._depth):
            self._w_h += [
                tf.get_variable(
                    "w_h" + str(l),
                    shape=[
                        self._h,
                        self._h],
                    initializer=tf.contrib.layers.xavier_initializer(
                        uniform=False))]
            self._b_h += [tf.get_variable("b_h" + str(l),
                                          shape=[self._h],
                                          initializer=tf.zeros_initializer())]
            self._w_m += [
                tf.get_variable(
                    "w_m" + str(l),
                    shape=[
                         self._m,
                         self._m],
                    initializer=tf.contrib.layers.xavier_initializer(
                        uniform=False))]
            self._b_m += [tf.get_variable("b_m" + str(l),
                                          shape=[self._m],
                                          initializer=tf.zeros_initializer())]
            self._w_up += [
                tf.get_variable(
                    "w_up" + str(l),
                    shape=[
                        self._h,
                        self._m],
                    initializer=tf.contrib.layers.xavier_initializer(
                        uniform=False))]
            self._b_up += [tf.get_variable("b_up" + str(l),
                                           shape=[self._m],
                                           initializer=tf.zeros_initializer())]
        self._w_x_h = tf.get_variable(
            "w_x_h",
            shape=[
                self._n,
                self._h],
            initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        self._b_x_h = tf.get_variable(
            "b_x_h", shape=[
                self._h], initializer=tf.zeros_initializer())

        self._w_x_m = tf.get_variable(
            "w_x_m",
            shape=[
                self._n,
                self._m],
            initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        self._b_x_m = tf.get_variable(
            "b_x_m", shape=[
                self._m], initializer=tf.zeros_initializer())

        self._w_s = tf.get_variable(
            "w_s",
            shape=[
                self._m,
                self._h],
            initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        self._b_s = tf.get_variable(
            "b_s", shape=[
                self._h], initializer=tf.zeros_initializer())
        self._processed_input = process_batch(self._inputs)

  def _network_gates(self, w, s):
        h = tf.tanh(w)
        t = tf.sigmoid(w)
        c = 1 - t
        t = tf.nn.dropout(t, self._drop)
        return h * t + s * c

  def _upscale(self, s):
        out_up = []
        for l in range(self._depth):
            y = tf.matmul(s[l], self._w_up[l]) + self._b_up[l]
            if self.batch_norm:
                batch_mean2, batch_var2 = tf.nn.moments(y, [0])
                y = tf.nn.batch_normalization(
                    y, batch_mean2, batch_var2, 1e-1, None, 1e-2)
            #out_up += [tf.concat([y,y], 1)]
            out_up += [y]
        return out_up

  def _hypernet(self, x, s):
        s_ = tf.matmul(s, self._w_s) + self._b_s			
        out_h = []
        for l in range(self._depth):
            w = tf.matmul(s_, self._w_h[l]) + self._b_h[l]
            if l == 0:
                u = tf.matmul(x, self._w_x_h) + self._b_x_h
                w += u
            s_ = self._network_gates(w, s_)
            if self.batch_norm:
                batch_mean2, batch_var2 = tf.nn.moments(s_, [0])
                s_ = tf.nn.batch_normalization(
                    s_, batch_mean2, batch_var2, 1e-5, None, 1e-5)
            out_h += [s_]
        return s, out_h

  def _main_net(self, x, s, z):
        out_m = []
        for l in range(self._depth):
            w = tf.matmul(s, self._w_m[l]) * z[l] + self._b_m[l]
            if l == 0:
                u = tf.matmul(x, self._w_x_m) * z[l] + self._b_x_m
                w += u
            s = self._network_gates(w, s)
            if self.batch_norm:
                batch_mean2, batch_var2 = tf.nn.moments(s, [0])
                s = tf.nn.batch_normalization(
                    s, batch_mean2, batch_var2, 1e-5, None, 1e-5)
            out_m += [s]
        return s, out_m

  def _do_some_work(self, s, x):
        _, out = self._hypernet(x, s)
        z = self._upscale(out)
        s_out, out = self._main_net(x, s, z)
        batch_mean2, batch_var2 = tf.nn.moments(s_out, [0])
        s_out = tf.nn.batch_normalization(
            s_out, batch_mean2, batch_var2, 1e-5, None, 1e-5)
        return s_out

  def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """
        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self._do_some_work,
                                    self._processed_input,
                                    initializer=self._initial_hidden,
                                    name='states')
        return all_hidden_states


def process_batch(batch_input):
       """
       from [batch_size, timesteps, features] to [timesteps, batch_size, features]
       """
    timestep = tf.transpose(batch_input, perm=[1, 0, 2])
    return timestep
