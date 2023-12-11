import argparse
import datetime
from math import ceil

import tensorflow as tf

import model as m
from edl_uncertainty import EDLCallback, EDLLoss, draw_EDL_results
import edl_uncertainty


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Max number of epochs.", type=int,
                    default=100)
parser.add_argument("--batch_size", help="Size of batch.", type=int, default=128)
parser.add_argument("--hidden_units", help="N of hidden neurons.", type=int,
                    default=100)
parser.add_argument("--train_mode", help="Either 'EDL' or 'CP'", default='CP')
parser.add_argument("--model", help="Model type. Chose from: "
                                    "Soli, NMNIST, SHD, DVS", default='Soli')

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
hidden_units = args.hidden_units
train_mode = args.train_mode
model = getattr(m, f'{args.model}Model')()

run_train_end = False
run_epoch_end = False
val = 'test'
if train_mode == 'EDL':
    val = 'cal'
    run_train_end = True
    run_epoch_end = False

model.load_data(cal=True, cal_portion=0.2)

edl_uncertainty.K = model.n_classes

callback_instance = None
if train_mode == 'EDL':
    model.preprocess(sparse_labels=False)
    n_batch = ceil(len(model.data['train']) / batch_size)
    callback_instance = EDLCallback(
        (model.data['train'], model.labels['train']),
        (model.data['test'], model.labels['test']),
        run_epoch_end=run_epoch_end,
        run_train_end=run_train_end,
        anneal=30*n_batch)
    model.build_edl_model(EDLLoss(callback_instance), hidden_units=hidden_units,
                          in_dropout=0.3)
elif train_mode == 'CP':
    model.preprocess(sparse_labels=True)
    model.build_model(hidden_units=hidden_units, in_dropout=0.2)
model.create_tf_dataset()
model.batch_tf(batch_size=batch_size)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10,
                                                  monitor='val_loss',
                                                  restore_best_weights=True)

model.fit(epochs=epochs, logging=False, val=val,
          callbacks=[callback_instance, early_stopping])
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.calc_score('cal')
model.calc_score('test')
model.save_scores(f'scores/scores_{train_mode}_{type(model).__name__}_{date}')
model.save(f'models/{train_mode}_{type(model).__name__}_{date}')

if train_mode == 'EDL' and run_train_end:
    draw_EDL_results(callback_instance.Seq_train_acc1,
                     callback_instance.Seq_train_ev_s,
                     callback_instance.Seq_train_ev_f,
                     callback_instance.Seq_test_acc1,
                     callback_instance.Seq_test_ev_s,
                     callback_instance.Seq_test_ev_f,
                     x_label='Frames')

print('done')
