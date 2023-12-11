import numpy as np
import datetime


class ConformalPrediction:
    def __init__(self, scores: np.ndarray, labels: np.ndarray, alpha: float):
        """

        :param scores: Output scores of model for calibration dataset.
                       2D array with shape (time_bins, n_classes)
        :param labels: One-hot encoded labels for calibration dataset.
        :param alpha: a-priori error-rate
        """
        self.scores = scores
        self.labels = labels

        self.alpha = alpha
        self.E = None
        self.q_hat = None

    def save_params(self):
        np.savez_compressed(
            f'conformal_params_'
            f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
            scores=self.scores,
            labels=self.labels,
            alpha=self.alpha,
            E=self.E,
            q_hat=self.q_hat
        )

    def load_params(self, params_npz: str):
        loaded = np.load(params_npz)
        self.scores = loaded['scores']
        self.labels = loaded['labels']
        self.alpha = loaded['alpha']
        self.E = loaded['E']
        self.q_hat = loaded['q_hat']

    def print_params(self):
        print(f'{self.alpha=}, {self.q_hat=}')

    def change_alpha(self, alpha):
        self._calc_q_hat(alpha)

    def _calc_q_hat(self, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        n_samples = len(self.scores)
        # finite sample effect correction
        quantile_val = np.ceil((n_samples + 1) * (1 - self.alpha)) / n_samples
        try:
            self.q_hat = np.quantile(self.E, quantile_val, method='higher')
        except TypeError:
            self.q_hat = np.quantile(self.E, quantile_val,
                                     interpolation='higher')

    def fit(self):
        e = []
        for score_label in zip(self.scores, self.labels):
            score, label = score_label
            # sorts scores descending order
            sorted_score, sorted_label = zip(*sorted(zip(score, label),
                                                     reverse=True))
            e_i = 0
            # sums up scores until true class is reached
            for i, score_i in enumerate(sorted_score):
                e_i += score_i
                if sorted_label[i]:
                    if e_i > 1:  # can surpass 1 due to rounding error
                        e_i = 1
                    e.append(e_i)
                    break
        self.E = np.array(e)
        self._calc_q_hat()
        return

    def predict(self, scores: np.ndarray, allow_empty=True):
        if self.q_hat is None:
            raise ValueError('Conformal Prediction needs to be'
                             'fit on calibration data first.')

        pred_set = []
        for score in scores:  # loops over all outputs
            tau_set = []
            summed = 0
            idx = np.arange(len(score))
            # sorts scores descending order
            sorted_score, sorted_idx = zip(*sorted(zip(score, idx),
                                                   reverse=True)
                                           )
            # loops over scores of output and sums up until over q_hat
            for i, score_i in enumerate(sorted_score):
                summed += score_i
                if summed >= self.q_hat:
                    # will lead to over-coverage but at inference time
                    # at least top-1 output shall be given
                    if not (allow_empty or len(tau_set) > 0):
                        tau_set.append((score_i, sorted_idx[i]))
                    pred_set.append(tau_set)
                    break
                tau_set.append((score_i, sorted_idx[i]))
                # if q_hat close to or equal 1 no pred_set may be appended
                # due to rounding error such that sum never reaches q_hat
                if i == len(sorted_score) - 1:  # reached when nothing appended
                    # print(f'Debug: No set was appended in {n}th iteration.')
                    pred_set.append(tau_set)
        return pred_set


class MovAvgConformalPrediction(ConformalPrediction):
    def __init__(self, use_smooth_len=True, use_comb_set=True, frames=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pred_hist = []
        self.len_hist = []
        self.smooth_len_hist = []
        self.pred_consist_hist = []
        self.smooth_len = None
        self.i_uc = 0
        self.test_scores = None
        self.test_labels = None
        self.current_len = None
        self.current_i = None

        self.use_smooth_len = use_smooth_len
        self.use_comb_set = use_comb_set
        if frames is None:
            raise ValueError('give the correct frame number!!!')
        self.frames = frames

    def predict(self, scores: np.ndarray, labels: np.ndarray = None,
                allow_empty=True):
        self.test_scores = scores
        self.test_labels = labels
        pred_set = []
        for i, s in enumerate(scores):
            if i % self.frames == 0:
                self.smooth_len = None
                self.pred_hist = []
            pred_set.append(super().predict(s[np.newaxis, :],
                                            allow_empty=allow_empty)[0])
            if pred_set[-1]:
                pred_classes = [x for x in zip(*pred_set[-1])][1]
            else:
                pred_classes = []

            self.current_len = len(pred_classes)
            self.current_i = i

            self.pred_hist.append(pred_classes)
            self.check_time_uncertain()

            self.len_hist.append(len(self.pred_hist[-1]))
            self.smooth_len_hist.append(self.smooth_len)
            self.pred_consist_hist.append(pred_classes)
        return pred_set

    def _calc_smoothed_len(self, alpha=0.6):
        try:
            # if not at start of frame and current length is greater than past
            if not self.current_i % self.frames == 0 \
                    and self.current_len > len(self.pred_hist[-2]) \
                    and self.use_smooth_len:
                smooth_len = self.current_len + self.smooth_len
            elif self.use_smooth_len:
                smooth_len = alpha * len(self.pred_hist[-1]) + (
                            1 - alpha) * self.smooth_len
            else:
                smooth_len = len(self.pred_hist[-1])
        except TypeError:  # multiplying with None at start
            smooth_len = len(self.pred_hist[-1])
        self.smooth_len = smooth_len

    def _calc_set_overlap(self, n_rec=1):
        current_set = self.pred_hist[-1]
        try:
            rec_set = self.pred_hist[-1-n_rec]
            # set not repeating elements
            combined_set = {*current_set, *rec_set}
            # if combined set larger current
            # and current and past set is same size
            if len(combined_set) > len(current_set) \
                    and (len(current_set) == len(self.pred_hist[-2])) \
                    and self.use_comb_set:
                self.i_uc += 1
                self.smooth_len += abs(len(combined_set) - len(current_set))
                # print(f'{self.i_uc}: Uncertainty in time found!')
        except IndexError:
            pass

    def check_time_uncertain(self):
        self._calc_smoothed_len()
        self._calc_set_overlap()
