import argparse
from sys import exit

import numpy as np
import matplotlib.pyplot as plt

from cp_uncertainty import MovAvgConformalPrediction
from cp_utils import get_coverage_list, acc_vs_len
from cp_utils import mean_len_against_frames
from utils import write_acc_uc, one_hot, str2bool


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model type. Chose from: "
                                    "Soli, NMNIST, SHD, DVS", default='Soli')
parser.add_argument("--allow_empty",
                    type=str2bool, nargs='?',
                    const=True,
                    help="Specify if sets can be empty."
                         "Coverage will be given but script"
                         "will terminate early", default=False)

ids = {'Soli': '20230724-102345',
       'NMNIST': '20230726-183648',
       'SHD': '20230727-090201',
       'DVS': '20230726-144948'
       }

args = parser.parse_args()
model = args.model
m_id = ids[model]  # if new models are trained, change the ID
allow_empty = args.allow_empty
model = f'{model}Model'

data_dict = {}

with np.load(f'scores/scores_CP_{model}_{m_id}.npz') as loaded:
    for key in loaded.files:
        data_dict[key] = loaded[key]

assert len(data_dict['cal_labels']) == len(data_dict['cal_scores'])

a_er = (1 - data_dict['acc']) * 0.5

data_dict['cal_labels'] = one_hot(data_dict['cal_labels'])
data_dict['test_labels'] = one_hot(data_dict['test_labels'])
labels = data_dict['test_labels']
CP = MovAvgConformalPrediction(scores=data_dict['cal_scores'],
                               labels=data_dict['cal_labels'],
                               alpha=a_er,
                               use_smooth_len=True,
                               use_comb_set=True,
                               frames=int(data_dict['frames']))
CP.fit()

predicted_sets = CP.predict(data_dict['test_scores'], labels,
                            allow_empty=allow_empty)

sum_tot = 0
set_len = []
for pred_set, label in zip(predicted_sets, labels):
    if len(pred_set) != 0:  # empty set deflates mean set size
        set_len.append(len(pred_set))
    for pred in pred_set:
        if pred[1] == np.argmax(label):
            sum_tot += 1
print(f'Set a-priori error-rate is: {100*a_er:.2f}%')
print(f'Average Set Length is: {np.mean(set_len):.2f}')
if allow_empty:
    print(f'Coverage is {100*sum_tot/len(labels):.2f}%')
    print(f'Actual error is {100*(1 - sum_tot/len(labels)):.2f}% '
          '(Table 3)')
    print('Exiting here since further experiments do not work with empty sets.')
    exit()

cov = get_coverage_list(CP.pred_consist_hist, CP.test_labels)
plt.figure()
plt.title(f'{model} (Zoom to closer resemble Figure 5)')
plt.plot(CP.len_hist, label='CP')
plt.plot(CP.smooth_len_hist, label='sCP')
plt.plot(cov, label='Coverage')
plt.xlabel('Decision Updates')
plt.ylabel('Set Length / Coverage')
plt.legend()
plt.show()

result_dict = acc_vs_len(CP)
print("Length for CP and sCP (s) for wrong (w) and correct (c) predictions "
      "(Table 2).")
print(result_dict)

acc, len_c, len_w = mean_len_against_frames(CP, CP.len_hist)
_, slen_c, slen_w = mean_len_against_frames(CP, CP.smooth_len_hist)
len_w = len_w / CP.labels.shape[-1]
len_c = len_c / CP.labels.shape[-1]
slen_w = slen_w / CP.labels.shape[-1]
slen_c = slen_c / CP.labels.shape[-1]

plt.figure()
plt.title(f'{model} (Figure 4)')
plt.plot(len_c[0], 'b-', label='CP_correct')
plt.plot(len_w[0], 'b--', label='CP_wrong')
plt.plot(slen_c[0], 'k-', label='sCP_correct')
plt.plot(slen_w[0], 'k--', label='sCP_wrong')
plt.plot(acc, 'b:', label='acc')
plt.xlabel('Frames')
plt.ylabel('Uncertainty / Accuracy')
plt.legend()
plt.show()

write_acc_uc(f'results/CP_{model}_{m_id}', acc=acc, uc_c=len_c[0],
             uc_w=len_w[0])
write_acc_uc(f'results/CP_{model}_{m_id}', suc_c=slen_c[0],
             suc_w=slen_w[0])
print('done')
