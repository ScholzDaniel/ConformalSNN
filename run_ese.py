import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ensemble import Ensemble
import model as m
from cp_utils import mean_softmax_against_frames, reduce_wrong_correct
from utils import write_acc_uc


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="Size of batch.", type=int,
                    default=128)
parser.add_argument("--path", help="Path to models.", default="models/ensemble")
parser.add_argument("--model", help="Model type. Chose from: "
                                    "Soli, NMNIST, SHD, DVS", default='Soli')


ids = {'Soli': '20230724-104957',
       'NMNIST': '20230726-190512',
       'SHD': '20230727-090314',
       'DVS': '20230727-095554'
       }

parser.add_argument("--n", help="Number of models in ensemble.", default=5)

args = parser.parse_args()
batch_size = args.batch_size
path = Path(args.path)
m_id = ids[args.model]  # if new models are trained, change the ID
n = args.n
model_class = getattr(m, f'{args.model}Model')

model = model_class()
name = type(model).__name__

model.load_data(cal=True, cal_portion=0.2)
model.preprocess(sparse_labels=True)
model.create_tf_dataset()
model.batch_tf(batch_size=batch_size)
ensemble = Ensemble()

for i in range(n):
    e_model = model_class()
    e_model.load_model(path/f'{i}_{name}_{m_id}')
    ensemble.add_model(e_model)

ensemble.predict(model.data['test'])
y_hat_mean = ensemble.decision()
frames = model.data['test'].shape[1]
match, uc = mean_softmax_against_frames(y_hat_mean, model.labels['test'],
                                        frames)
acc = np.mean(match, axis=0)
smax_c, smax_w = reduce_wrong_correct(match, uc)

gry_scl = 0.7
plt.figure()
plt.title(f'{name} (Figure 4)')
plt.plot(smax_c[0], '-', color=(gry_scl, gry_scl, gry_scl), label='ESE_correct')
plt.plot(smax_w[0], '--', color=(gry_scl, gry_scl, gry_scl), label='ESE_wrong')
plt.plot(acc, ':', color=(gry_scl, gry_scl, gry_scl), label='acc')
plt.xlabel('Frames')
plt.ylabel('Uncertainty / Accuracy')
plt.legend()
plt.show()

write_acc_uc(f'results/ESE_{type(model).__name__}_{m_id}', acc=acc,
             uc_c=smax_c[0],
             uc_w=smax_w[0])
print('done')
