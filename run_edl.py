import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import model as m
from edl_uncertainty import LossLoad
from edl_uncertainty import edl_metrics
from utils import write_acc_uc


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="Size of batch.", type=int,
                    default=128)
parser.add_argument("--model", help="Model type. Chose from: "
                                    "Soli, NMNIST, SHD, DVS", default='Soli')

ids = {'Soli': '20230724-112321',
       'NMNIST': '20230726-224647',
       'SHD': '20230727-092211',
       'DVS': '20230727-101143'
       }

args = parser.parse_args()
batch_size = args.batch_size
m_id = ids[args.model]  # if new models are trained, change the ID
model = getattr(m, f'{args.model}Model')()

name = type(model).__name__

model.load_data(cal=True, cal_portion=0.2)

model.preprocess(sparse_labels=False)
model.create_tf_dataset()
model.batch_tf(batch_size=batch_size)
model.load_model(f'models/EDL_{name}_{m_id}', loss_func=LossLoad,
                 custom_loss=True)

Seq_acc = []
Seq_ev_s = []
Seq_ev_f = []
preds = model.predict(model.data['test'])[1]
for i in range(model.data['test'].shape[1]):
    pred = preds[:, i, tf.newaxis, :]
    acc, mean_ev_succ, mean_ev_fail = edl_metrics(model.labels['test'], pred)
    Seq_acc.append(acc)
    Seq_ev_s.append(mean_ev_succ)
    Seq_ev_f.append(mean_ev_fail)
K = model.n_classes
test_u_succ = K / (K + np.array(Seq_ev_s))
test_u_fail = K / (K + np.array(Seq_ev_f))

plt.figure()
plt.title(f'{name} (Figure 4)')
plt.plot(test_u_succ, 'r-', label='correct')
plt.plot(test_u_fail, 'r--', label='wrong')
plt.plot(Seq_acc, 'r:', label='acc')
plt.xlabel('Frames')
plt.ylabel('Uncertainty / Accuracy')
plt.legend()
plt.show()

write_acc_uc(f'results/EDL_{type(model).__name__}_{m_id}', acc=Seq_acc,
             uc_c=test_u_succ,
             uc_w=test_u_fail)

print('done')
