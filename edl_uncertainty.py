"""
This code is based on the code from
https://github.com/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb
but reimplemented in TensorFlow 2.

We implemented the original author's MNIST experiments to validate our
implementation. This file can be executed, mnist_example() will serve as main
function.
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from gpu_selection import select_gpu
select_gpu(0)


def rotate_img(x, deg):
    import scipy.ndimage as nd
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()


# This function to generate evidence is used for the first example
def relu_evidence(logits):
    return tf.nn.relu(logits)


# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits):
    return tf.exp(tf.clip_by_value(logits/10, -10, 10))


# This one is another alternative and
# usually behaves better than the relu_evidence
def softplus_evidence(logits):
    return tf.nn.softplus(logits)


# This method rotates an image counter-clockwise and classify it for different degress of rotation.
# It plots the highest classification probability along with the class label for each rotation degree.
def rotating_image_classification(img, model, uncertainty=True, threshold=0.5):
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    scores = np.zeros((1, K))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img, deg).reshape(28, 28)
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        rimgs[:, i * 28:(i + 1) * 28] = nimg
        if not uncertainty:
            _, p_pred_t = edl_uncertainty(model(nimg.reshape((1, 28, 28, 1))))
        else:
            u, p_pred_t = edl_uncertainty(model(nimg.reshape((1, 28, 28, 1))))
            lu.append(u.numpy().mean())
        scores += p_pred_t.numpy() >= threshold
        ldeg.append(deg)
        lp.append(p_pred_t[0].numpy())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ['black', 'blue', 'red', 'brown', 'purple', 'cyan']
    marker = ['s', '^', 'o'] * 2
    labels = labels.tolist()
    for i in range(len(labels)):
        plt.plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if uncertainty is not None:
        labels += ['uncertainty']
        plt.plot(ldeg, lu, marker='<', c='red')

    plt.legend(labels)

    plt.xlim([0, Mdeg])
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    plt.figure(figsize=[6.2, 100])
    plt.imshow(1 - rimgs, cmap='gray')
    plt.axis('off')
    plt.show()


def KL(alpha):
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=-1, keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),
                                                  axis=-1,
                                                  keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=-1,
                            keepdims=True) - tf.math.lgamma(S_beta)

    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)

    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=-1,
                       keepdims=True) + lnB + lnB_uni
    return kl


def edl_frames(model, callback, set='train'):
    preds = model.predict(getattr(callback, f'{set}_data')[0])[1]
    for i in range(callback.frames):
        pred = preds[:, i, tf.newaxis, :]
        acc, mean_ev_succ, mean_ev_fail = edl_metrics(
            getattr(callback, f'{set}_data')[1], pred)
        getattr(callback, f'Seq_{set}_acc1').append(acc)
        getattr(callback, f'Seq_{set}_ev_s').append(mean_ev_succ)
        getattr(callback, f'Seq_{set}_ev_f').append(mean_ev_fail)
    return 0


class EDLCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, test_data, run_epoch_end=False,
                 run_train_end=False, anneal=500, mnist=False):
        self.train_data = train_data
        self.test_data = test_data
        self.frames = train_data[0].shape[1]

        self.gl_step = tf.Variable(0.0)
        self.aneal_step = tf.Variable(anneal)

        self.L_train_acc1 = []
        self.L_train_ev_s = []
        self.L_train_ev_f = []
        self.L_test_acc1 = []
        self.L_test_ev_s = []
        self.L_test_ev_f = []
        self.Seq_train_acc1 = []
        self.Seq_train_ev_s = []
        self.Seq_train_ev_f = []
        self.Seq_test_acc1 = []
        self.Seq_test_ev_s = []
        self.Seq_test_ev_f = []
        self.run_epoch_end = run_epoch_end
        self.run_train_end = run_train_end
        self.mnist = mnist

    def on_batch_end(self, batch, logs=None):
        self.gl_step.assign_add(1)
        # print(f'\r{self.gl_step=}', end='')

    def on_epoch_end(self, batch, logs=None):
        if not self.run_epoch_end:
            return
        if self.mnist:
            acc, mean_ev_succ, mean_ev_fail = edl_metrics(self.train_data[1],
                                                          self.model.predict(
                                                              self.train_data[0]
                                                          ))
        else:
            acc, mean_ev_succ, mean_ev_fail = edl_metrics(
                self.train_data[1], self.model.predict(self.train_data[0])[0]
            )
        self.L_train_acc1.append(acc)
        self.L_train_ev_s.append(mean_ev_succ)
        self.L_train_ev_f.append(mean_ev_fail)

        if self.mnist:
            acc, mean_ev_succ, mean_ev_fail = edl_metrics(
                self.test_data[1], self.model.predict(self.test_data[0])
            )
        else:
            acc, mean_ev_succ, mean_ev_fail = edl_metrics(
                self.test_data[1], self.model.predict(self.test_data[0])[0]
            )
        self.L_test_acc1.append(acc)
        self.L_test_ev_s.append(mean_ev_succ)
        self.L_test_ev_f.append(mean_ev_fail)

    def on_train_end(self, batch, logs=None):
        if not self.run_train_end:
            return
        edl_frames(self.model, self, set='train')
        edl_frames(self.model, self, set='test')


def edl_uncertainty(y_pred):
    evidence = relu_evidence(y_pred)
    alpha = evidence + 1

    u = K / tf.math.reduce_sum(alpha, axis=-1, keepdims=True)  # uncertainty
    prob = alpha / tf.math.reduce_sum(alpha, 1, keepdims=True)

    return u, prob


def edl_metrics(y_true, y_pred):
    evidence = relu_evidence(y_pred)
    y_true = tf.cast(y_true, dtype=tf.float32)
    pred = tf.argmax(y_pred, axis=-1)
    truth = tf.argmax(y_true, axis=-1)
    if len(evidence.shape) == 3:
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),
                           (-1, 1, 1))
    else:
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),
                           (-1, 1))
    acc = tf.reduce_mean(match)

    total_evidence = tf.reduce_sum(evidence, axis=-1, keepdims=True)
    mean_ev = tf.reduce_mean(total_evidence)
    mean_ev_succ = tf.reduce_sum(
        tf.reduce_sum(
            evidence, axis=-1,
            keepdims=True
        ) * match) / tf.reduce_sum(match + 1e-20)
    mean_ev_fail = tf.reduce_sum(
        tf.reduce_sum(
            evidence, axis=-1,
            keepdims=True
        ) * (1 - match)) / (tf.reduce_sum(tf.abs(1 - match)) + 1e-20)
    return acc, mean_ev_succ, mean_ev_fail


def edl_exp_cross(func=tf.math.digamma):
    def loss_func(y_true, alpha, callback):
        loss = edl_expected_crossentropy(y_true, alpha, callback.gl_step,
                                         callback.aneal_step, func)
        return loss
    return loss_func


def edl_expected_crossentropy(p, alpha, global_step, annealing_step, func):
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    E = alpha - 1

    A = tf.reduce_sum(p * (func(S) - func(alpha)), axis=-1, keepdims=True)

    annealing_coef = tf.minimum(1.0,
                                tf.divide(global_step,
                                          tf.cast(annealing_step,
                                                  dtype=tf.float32))
                                )

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp)

    return (A + B)


def edl_loss(y_true, alpha, callback):
    loss = tf.reduce_mean(
        mse_loss(y_true, alpha, callback.gl_step, callback.aneal_step)
    )
    return loss


def mse_loss(p, alpha, global_step, annealing_step):
    S = tf.math.reduce_sum(alpha, axis=-1, keepdims=True)
    E = alpha - 1
    m = alpha / S

    A = tf.math.reduce_sum((p - m) ** 2, axis=-1, keepdims=True)
    B = tf.math.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=-1,
                           keepdims=True)

    # annealing_coef = tf.minimum(1.0, tf.cast(global_step / annealing_step,
    #                                          tf.float32))
    annealing_coef = tf.minimum(
        1.0,
        tf.divide(global_step, tf.cast(annealing_step, dtype=tf.float32))
    )

    alp = E * (1 - p) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C


class LossLoad(tf.keras.losses.Loss):
    def __init__(self, reduction='auto', name=None):
        super().__init__()

    def call(self, y_true, y_pred):
        return loss_load(y_true, y_pred)


def loss_load(y_true, y_pred):
    evidence = relu_evidence(y_pred)
    alpha = evidence + 1
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = tf.reduce_mean(mse_loss(y_true, alpha, 1, 1))
    return loss


class EDLLoss(tf.keras.losses.Loss):
    def __init__(self, callback=None, ev_func=relu_evidence, loss_func=edl_loss,
                 reduction='auto', name=None):
        # reduction and auto needed here to load model with custom loss
        # always pass an EDLCallback for the anneal and global step
        super().__init__()
        self.ev_func = ev_func
        self.loss_func = loss_func
        self.callback = callback

    def call(self, y_true, y_pred):
        evidence = self.ev_func(y_pred)
        alpha = evidence + 1
        y_true = tf.cast(y_true, dtype=tf.float32)
        return self.loss_func(y_true, alpha, self.callback)


def model(callback):
    x_in = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(20, (5, 5), activation='relu')(x_in)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(50, (5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    regularizer = tf.keras.regularizers.L2(l2=0.005)
    x_out = tf.keras.layers.Dense(10, kernel_regularizer=regularizer,
                                  name='output')(x)
    model = tf.keras.models.Model(inputs=[x_in], outputs=[x_out])

    model.compile(loss=EDLLoss(callback, ev_func=exp_evidence,
                               loss_func=edl_exp_cross(tf.math.log)),
                  optimizer="adam",
                  metrics='accuracy',
                  run_eagerly=False)

    shape = (28, 28, 1)
    model.build(input_shape=shape)
    model.summary()
    return model


def draw_EDL_results(train_acc1, train_ev_s, train_ev_f, test_acc1, test_ev_s,
                     test_ev_f, x_label='Epoch'):
    # calculate uncertainty for training and testing data
    # for correctly and misclassified samples
    train_u_succ = K / (K + np.array(train_ev_s))
    train_u_fail = K / (K + np.array(train_ev_f))
    test_u_succ = K / (K + np.array(test_ev_s))
    test_u_fail = K / (K + np.array(test_ev_f))

    f, axs = plt.subplots(2, 2)
    f.set_size_inches([10, 10])

    axs[0, 0].plot(train_ev_s, c='r', marker='+')
    axs[0, 0].plot(train_ev_f, c='k', marker='x')
    axs[0, 0].set_title('Train Data')
    axs[0, 0].set_xlabel(x_label)
    axs[0, 0].set_ylabel('Estimated total evidence for classification')
    axs[0, 0].legend(['Correct classifications', 'Misclassifications'])

    axs[0, 1].plot(train_u_succ, c='r', marker='+')
    axs[0, 1].plot(train_u_fail, c='k', marker='x')
    axs[0, 1].plot(train_acc1, c='blue', marker='*')
    axs[0, 1].set_title('Train Data')
    axs[0, 1].set_xlabel(x_label)
    axs[0, 1].set_ylabel('Estimated uncertainty for classification')
    axs[0, 1].legend(
        ['Correct classifications', 'Misclassifications', 'Accuracy'])

    axs[1, 0].plot(test_ev_s, c='r', marker='+')
    axs[1, 0].plot(test_ev_f, c='k', marker='x')
    axs[1, 0].set_title('Test Data')
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel('Estimated total evidence for classification')
    axs[1, 0].legend(['Correct classifications', 'Misclassifications'])

    axs[1, 1].plot(test_u_succ, c='r', marker='+')
    axs[1, 1].plot(test_u_fail, c='k', marker='x')
    axs[1, 1].plot(test_acc1, c='blue', marker='*')
    axs[1, 1].set_title('Test Data')
    axs[1, 1].set_xlabel(x_label)
    axs[1, 1].set_ylabel('Estimated uncertainty for classification')
    axs[1, 1].legend(
        ['Correct classifications', 'Misclassifications', 'Accuracy'])

    plt.show()


def mnist_example():
    """
    Reproducing results from the original author's code in Tensorflow 1
    Original TensorFlow 1 Code from:
    https://github.com/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb
    :return: None
    """
    global K
    K = 10
    batch_size = 100
    epochs = 10

    (train_images, train_labels), (test_images, test_labels) = tf.keras.\
        datasets.mnist.load_data(path='/data/Datasets/mnist/mnist.npz')
    train_images = train_images / 255
    test_images = test_images / 255
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    digit_one = train_images[6].copy()

    from utils import one_hot
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)
    from math import ceil
    anneal = ceil(10 * (len(train_images) / batch_size))
    callback_instance = EDLCallback((train_images, train_labels),
                                    (test_images, test_labels), mnist=True,
                                    run_epoch_end=True,
                                    anneal=anneal)
    m = model(callback_instance)
    m.fit(train_images, train_labels,
          validation_data=(test_images, test_labels), epochs=epochs,
          callbacks=[callback_instance], batch_size=batch_size)
    draw_EDL_results(callback_instance.L_train_acc1,
                     callback_instance.L_train_ev_s,
                     callback_instance.L_train_ev_f,
                     callback_instance.L_test_acc1,
                     callback_instance.L_test_ev_s,
                     callback_instance.L_test_ev_f)
    rotating_image_classification(digit_one, m)


if __name__ == '__main__':
    mnist_example()
    print('done')
