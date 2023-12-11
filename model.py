from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.summary import create_file_writer, scalar
from tensorflow.keras.models import Model
from tensorflow.data import Dataset

from utils import one_hot, unison_shuffle
from neuron import IntegratorCell, LIF
from gpu_selection import select_gpu
select_gpu(0)


def predict_and_stack(model, data):
    scores = [model(dat)[1].numpy() for dat, _ in data]
    scores_stacked = np.vstack(
        [score.reshape(-1, score.shape[-1]) for score in scores]
    )
    return scores_stacked


class BaseModel:
    def __init__(self):
        self.model = None
        self.data = {}
        self.labels = {}
        self.scores_labels = {}
        self.tf_dataset = {}

    def fit(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError


class FrameModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model: Model
        self.batch_size = None
        self.frames = None
        self.n_classes = None
        self.tf_callbacks = None

    def load_data(self):
        raise NotImplementedError

    def fit(self, val='cal', **kwargs):
        self.tf_callbacks = kwargs.setdefault('callbacks', [])
        self.tf_callbacks = [x for x in self.tf_callbacks if x is not None]
        if kwargs['logging']:
            self.tf_callbacks.append(self.init_logging())
        kwargs.pop('callbacks', None)
        kwargs.pop('logging', None)
        self.model.fit(self.tf_dataset['train'],
                       validation_data=self.tf_dataset[val],
                       callbacks=self.tf_callbacks,
                       **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def one_hot_labels(self):
        for k, v in self.labels.items():
            self.labels[k] = one_hot(v)

    def create_tf_dataset(self):
        dataset = {}
        for k, v in self.data.items():
            dataset[k] = Dataset.from_tensor_slices(
                (tf.convert_to_tensor(v, dtype=tf.uint8),
                 tf.convert_to_tensor(self.labels[k], dtype=tf.uint8))
            )
        dataset['train'] = dataset['train'].shuffle(
            len(dataset['train']),
            reshuffle_each_iteration=True)
        self.tf_dataset = dataset

    def batch_tf(self, batch_size=128):
        for k, v in self.tf_dataset.items():
            self.tf_dataset[k] = v.batch(batch_size)
        self.batch_size = batch_size

    def save_scores(self, file_name):
        loss, _, acc = self.model.evaluate(self.tf_dataset['test'])
        np.savez(file_name, **self.scores_labels, frames=self.frames, acc=acc)

    def calc_score(self, dataset: str):
        shape = self.data[dataset].shape
        frames = shape[1]
        self.frames = frames
        scores = predict_and_stack(self.model, self.tf_dataset[dataset])
        labels = [[lab[0]]*frames for lab in self.labels[dataset]]
        labels = np.array(labels).reshape(-1)[:, np.newaxis]
        self.scores_labels[dataset + '_scores'] = scores
        self.scores_labels[dataset + '_labels'] = labels
        print('done')

    def init_logging(self):
        logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = create_file_writer(logdir + "/params")
        file_writer.set_as_default()
        scalar('batch size', data=self.batch_size, step=1)
        # scalar('frames', data=self.frames, step=1)
        return TensorBoard(log_dir=logdir)

    def base_rnn(self, hidden_units=15, in_dropout=0.0):
        shape = self.data['train'].shape
        input_neurons = shape[-1]
        frames = shape[1]

        in_spikes = tf.keras.layers.Input(shape=(frames, input_neurons),
                                          name='input')

        dropout = tf.keras.layers.Dropout(in_dropout)(in_spikes)

        hidden, state_v = tf.keras.layers.RNN(
            LIF(n_in=in_spikes.shape[-1], n_rec=hidden_units, tau=10, thr=1,
                dt=1, recurrence=False),
            return_sequences=True, name='HL01')(dropout)

        dropout_hid = tf.keras.layers.Dropout(0.0)(hidden)

        output = tf.keras.layers.RNN(
            IntegratorCell(n_in=hidden_units, n_rec=self.n_classes),
            return_sequences=True,
            name='seq')(dropout_hid)

        output_z = tf.slice(output, [0, frames - 1, 0], [-1, 1, self.n_classes],
                            name='tf.slice')

        model = tf.keras.models.Model(inputs=[in_spikes],
                                      outputs=[output_z, output, hidden,
                                               state_v])
        out_name = output_z.name.split('/')[0]
        model.compile(loss={
            out_name: tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)},
            optimizer="adam",
            metrics={out_name: 'accuracy'},
            run_eagerly=False)

        shape = (frames, input_neurons)
        model.build(input_shape=shape)
        model.summary()
        return model

    def edl_model(self, loss_func, hidden_units=15, in_dropout=0.0):
        # regularizer = tf.keras.regularizers.L2(l2=0.005)
        regularizer = None
        shape = self.data['train'].shape
        input_neurons = shape[-1]
        frames = shape[1]

        in_spikes = tf.keras.layers.Input(shape=(frames, input_neurons),
                                          name='input')

        dropout = tf.keras.layers.Dropout(in_dropout)(in_spikes)

        hidden, state_v = tf.keras.layers.RNN(
            LIF(n_in=in_spikes.shape[-1], n_rec=hidden_units, tau=10, thr=1,
                dt=1, recurrence=False, regularizer=regularizer),
            return_sequences=True, name='HL01')(dropout)

        dropout_hid = tf.keras.layers.Dropout(0.0)(hidden)

        output = tf.keras.layers.RNN(
            IntegratorCell(n_in=hidden_units, n_rec=self.n_classes,
                           softmax=False, regularizer=regularizer),
            return_sequences=True,
            name='seq')(dropout_hid)

        output_z = tf.slice(output, [0, frames - 1, 0], [-1, 1, self.n_classes],
                            name='tf.slice')

        model = tf.keras.models.Model(inputs=[in_spikes],
                                      outputs=[output_z, output, hidden,
                                               state_v])
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss={
            "tf.slice": loss_func},
            optimizer=opt,
            metrics={'tf.slice': 'accuracy'},
            run_eagerly=False)

        shape = (frames, input_neurons)
        model.build(input_shape=shape)
        model.summary()
        return model

    def load_model(self, path, loss_func=None, custom_loss=False):
        if custom_loss:
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    'LIF': LIF,
                    'IntegratorCell': IntegratorCell,
                    'EDLLoss': loss_func,
                })
        else:
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    'LIF': LIF,
                    'IntegratorCell': IntegratorCell,
                    # 'EDLLoss': loss_func,
                })
        model.summary()
        self.model = model


class SoliModel(FrameModel):
    def __init__(self):
        super().__init__()
        self.n_classes = 11
        self.name = 'Soli'

    def load_data(self, cal=True, cal_portion=0.2):
        with np.load('data/soli_dataset_train_c11.npz') as loaded:
            train_data = loaded['dataset']
            train_labels = loaded['labels']
        with np.load('data/soli_dataset_test_c11.npz') as loaded:
            val_data = loaded['dataset']
            val_labels = loaded['labels']

        if cal:
            s = int((1-cal_portion) * len(train_data))
            self.data['cal'] = train_data[s:]
            self.labels['cal'] = train_labels[s:]
        else:
            s = len(train_data)
        self.data['train'] = train_data[:s]
        self.data['test'] = val_data
        self.labels['train'] = train_labels[:s]
        self.labels['test'] = val_labels

    @staticmethod
    def spike_encode(data, q=0):
        spike_data = np.where(data > q, 1, 0)
        return spike_data

    def preprocess(self, sparse_labels=True):
        if not self.data:
            raise ValueError('Load Data first!')

        for k, v in self.data.items():
            v = v.reshape(*v.shape[:-2], -1)
            self.data[k] = self.spike_encode(v)

        for k, v in self.labels.items():
            if sparse_labels:
                v = np.argmax(v, axis=1)
                self.labels[k] = np.expand_dims(v, axis=-1)
            else:
                self.labels[k] = np.expand_dims(v, axis=1)

    def build_model(self, hidden_units=15, in_dropout=0.0):
        self.model = self.base_rnn(hidden_units=hidden_units,
                                   in_dropout=in_dropout)

    def build_edl_model(self, loss_func, hidden_units=15, in_dropout=0.0):
        self.model = self.edl_model(loss_func, hidden_units=hidden_units,
                                    in_dropout=in_dropout)


class DVSModel(FrameModel):
    def __init__(self):
        super().__init__()
        self.n_classes = 11
        self.name = 'DVS'

    def load_data(self, cal=True, cal_portion=0.2):
        with np.load('data/dvs_29.npz') as loaded:
            train_data = loaded['x_train']
            train_labels = loaded['y_train']
            test_data = loaded['x_valid']
            test_labels = loaded['y_valid']
        if cal:
            s = int((1 - cal_portion) * len(train_data))
            self.data['cal'] = train_data[s:, :, ...]
            self.labels['cal'] = train_labels[s:]
        else:
            s = len(train_data)
        self.data['train'] = train_data[:s, :, ...]
        self.data['test'] = test_data
        self.labels['train'] = train_labels[:s]
        self.labels['test'] = test_labels

    @staticmethod
    def spike_encode(data, q=0):
        spike_data = np.where(data > q, 1, 0)
        return spike_data

    def preprocess(self, sparse_labels=True):
        if not self.data:
            raise ValueError('Load Data first!')

        for k, v in self.data.items():
            self.data[k] = self.spike_encode(v)

        for k, v in self.labels.items():
            if sparse_labels:
                self.labels[k] = np.expand_dims(v, axis=-1)
            else:
                v = one_hot(v)
                self.labels[k] = np.expand_dims(v, axis=1)

    def build_model(self, hidden_units=15, in_dropout=0.0):
        self.model = self.base_rnn(hidden_units=hidden_units,
                                   in_dropout=in_dropout)

    def build_edl_model(self, loss_func, hidden_units=15, in_dropout=0.0):
        self.model = self.edl_model(loss_func, hidden_units=hidden_units,
                                    in_dropout=in_dropout)


class SHDModel(FrameModel):
    def __init__(self):
        super().__init__()
        self.n_classes = 20
        self.name = 'SHD'

    def load_data(self, cal=True, cal_portion=0.2):
        with np.load('data/shd_20.npz') as loaded:
            train_data = loaded['x_train']
            train_labels = loaded['y_train']
            test_data = loaded['x_valid']
            test_labels = loaded['y_valid']
        if cal:
            s = int((1-cal_portion) * len(train_data))
            self.data['cal'] = train_data[s:, :, ...]
            self.labels['cal'] = train_labels[s:]
        else:
            s = len(train_data)
        self.data['train'] = train_data[:s, :, ...]
        self.data['test'] = test_data
        self.labels['train'] = train_labels[:s]
        self.labels['test'] = test_labels

    def preprocess(self, sparse_labels=True):
        if not self.data:
            raise ValueError('Load Data first!')

        for k, v in self.data.items():
            self.data[k] = self.spike_encode(v)

        for k, v in self.labels.items():
            if sparse_labels:
                self.labels[k] = np.expand_dims(v, axis=-1)
            else:
                v = one_hot(v)
                self.labels[k] = np.expand_dims(v, axis=1)

    @staticmethod
    def spike_encode(data, q=0):
        spike_data = np.where(data > q, 1, 0)
        return spike_data

    def build_model(self, hidden_units=15, in_dropout=0.0):
        self.model = self.base_rnn(hidden_units=hidden_units,
                                   in_dropout=in_dropout)

    def build_edl_model(self, loss_func, hidden_units=15, in_dropout=0.0):
        self.model = self.edl_model(loss_func, hidden_units=hidden_units,
                                    in_dropout=in_dropout)


class NMNISTModel(FrameModel):
    def __init__(self):
        super().__init__()
        self.n_classes = 10
        self.name = 'NMNIST'

    def load_data(self, cal=True, cal_portion=0.2):
        with np.load('data/nmnist_30.npz') as loaded:
            train_data = loaded['x_train']
            train_labels = loaded['y_train']
            test_data = loaded['x_valid']
            test_labels = loaded['y_valid']
        if cal:
            train_data, train_labels = unison_shuffle(train_data, train_labels)
            s = int((1-cal_portion) * len(train_data))
            self.data['cal'] = train_data[s:, :, ...]
            self.labels['cal'] = train_labels[s:]
        else:
            s = len(train_data)
        self.data['train'] = train_data[:s, :, ...]
        self.data['test'] = test_data
        self.labels['train'] = train_labels[:s]
        self.labels['test'] = test_labels

    def preprocess(self, sparse_labels=True):
        if not self.data:
            raise ValueError('Load Data first!')

        for k, v in self.data.items():
            self.data[k] = self.spike_encode(v).astype(np.uint8)

        for k, v in self.labels.items():
            if sparse_labels:
                self.labels[k] = np.expand_dims(v, axis=-1)
            else:
                v = one_hot(v)
                self.labels[k] = np.expand_dims(v, axis=1)

    @staticmethod
    def spike_encode(data, q=0):
        spike_data = np.where(data > q, 1, 0)
        return spike_data

    def build_model(self, hidden_units=15, in_dropout=0.0):
        self.model = self.base_rnn(hidden_units=hidden_units,
                                   in_dropout=in_dropout)

    def build_edl_model(self, loss_func, hidden_units=15, in_dropout=0.0):
        self.model = self.edl_model(loss_func, hidden_units=hidden_units,
                                    in_dropout=in_dropout)
