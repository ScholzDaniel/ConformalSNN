import numpy as np


class Ensemble:
    def __init__(self):
        self.models = []
        self.predictions = []

    def add_model(self, model):
        """
        Adds a model in the pool of models which the ensembles uses for
        predictions.

        :param model:
        :type model: model.FrameModel
        """
        self.models.append(model)

    def predict(self, data):
        """
        Predicts with each model and stores the output.

        :param data:
        :type data: np.ndarray
        """
        self.predictions = []
        for model in self.models:
            self.predictions.append(model.predict(data)[1])

    def decision(self):
        """
        Calculates the decision of the ensemble when predictions are available.
        :return: np.ndarray
        """
        if not self.predictions:
            raise ValueError("Run predict first.")
        arr = np.array(self.predictions)
        means = np.mean(arr, axis=0)
        return means
