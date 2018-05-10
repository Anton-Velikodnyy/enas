import numpy as np
import tensorflow as tf


def lstm(x, prev_c, prev_h, w):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def basic_neat(x, prev_c, prev_h, w):
    w_x, w_h = tf.split(w, 2, axis=0)
    x_w_x = tf.matmul(x, w_h)
    h_w_h = tf.matmul(prev_h, w_h)

    gates_h = tf.split(h_w_h, 8, axis=1)
    gates_x = tf.split(x_w_x, 8, axis=1)

    a = gates_x[0] * gates_h[0]
    b = gates_x[1] + gates_h[1]
    c = gates_x[2] + gates_h[2]
    d = gates_x[3] + gates_h[3]
    e = gates_x[4] + gates_h[4]
    f = gates_x[5] + gates_h[5]
    g = gates_x[6] + gates_h[6]
    h = gates_x[7] + gates_h[7]

    tanh = tf.tanh
    relu = tf.nn.relu
    sigmoid = tf.sigmoid
    identity = tf.identity

    # Nas cell 1
    next_c = tanh(prev_c + tanh(relu(h) * sigmoid(g))) * tanh(sigmoid(d) + relu(a))
    next_h = tanh(identity(next_c) * tanh(tanh(sigmoid(e) * tanh(f)) + sigmoid(sigmoid(b) + tanh(c))))
    return next_c, next_h


def stack_basic_neat(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = basic_neat(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def advanced_neat(x, prev_c, prev_h, w):
    w_x, w_h = tf.split(w, 2, axis=0)
    x_w_x = tf.matmul(x, w_h)
    h_w_h = tf.matmul(prev_h, w_h)
    gates_h = tf.split(h_w_h, 8, axis=1)
    gates_x = tf.split(x_w_x, 8, axis=1)
    a = gates_x[0] + gates_h[0]
    b = gates_x[1] + gates_h[1]
    c = gates_x[2] + gates_h[2]
    d = gates_x[3] + gates_h[3]
    e = gates_x[4] + gates_h[4]
    f = tf.maximum(gates_x[5], gates_h[5])
    g = tf.maximum(gates_x[6], gates_h[6])
    h = tf.maximum(gates_x[7], gates_h[7])

    tanh = tf.tanh
    relu = tf.nn.relu
    sigmoid = tf.sigmoid
    identity = tf.identity

    # Nas cell 2
    next_c = identity(identity(prev_c + tanh(h)) + identity(g)) * sigmoid(relu(a) + tanh(d))
    next_h = tanh(identity(next_c) * sigmoid(sigmoid(tanh(f) + tanh(e)) * sigmoid(identity(c) + tanh(b))))
    return next_c, next_h


def stack_advanced_neat(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = advanced_neat(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=initializer)
