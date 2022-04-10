import copy
import numpy as np


class Ensemble(object):
    def __init__(self):
        self.models = []

    def add_model(self, model, weight=1.0, transform=None):
        self.models.append({
            'model': model,
            'weight': weight,
            'transform': transform,
        })

    def predict(self, X):
        results = None
        weights = 0

        for imodel in self.models:
            imodel_mdl = imodel['model']
            imodel_tr = imodel['transform']
            iweight = imodel['weight']

            if imodel_tr is not None:
                Xc = copy.deepcopy(X)
                Xc, _ = imodel_tr(Xc)

            ires = imodel_mdl.predict(Xc)
            if not isinstance(ires, np.ndarray):
                ires = ires.to_numpy()

            iwres = iweight * ires

            if results is None:
                results = iwres

            else:
                results += iwres

            weights += iweight

        results = results / weights

        return results
