import argparse
import datetime

import tensorflow as tf

import model as m

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Max number of epochs.", type=int,
                    default=100)
parser.add_argument("--batch_size", help="Size of batch.", type=int,
                    default=128)
parser.add_argument("--hidden_units", help="N of hidden neurons.", type=int,
                    default=100)
parser.add_argument("--runs", help="Number of runs.", type=int, default=5)
parser.add_argument("--model", help="Model type. Chose from: "
                                    "Soli, NMNIST, SHD, DVS",
                    default='Soli')

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
hidden_units = args.hidden_units
runs = args.runs
model = getattr(m, f'{args.model}Model')()

model.load_data(cal=True, cal_portion=0.2)
model.preprocess(sparse_labels=True)
model.create_tf_dataset()
model.batch_tf(batch_size=batch_size)

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for i in range(runs):
    model.build_model(hidden_units=hidden_units, in_dropout=0.3)

    early_stopping = tf.keras.callbacks. \
        EarlyStopping(patience=10,
                      monitor='val_loss',
                      restore_best_weights=True)
    model.fit(epochs=epochs, logging=False,
              callbacks=[early_stopping], val='cal')
    model.save(f'models/ensemble/{i}_{type(model).__name__}_{date}')
