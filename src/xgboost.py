import graphlab as gl
import numpy as np
import os

from sklearn.base import BaseEstimator
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import utility as utils

class XGBoost(BaseEstimator):
    def __init__(self, max_iterations=50, max_depth=9, min_child_weight=4, row_subsample=.75,
                 min_loss_reduction=1., column_subsample=.8, step_size=.3, verbose=True):
        self.n_classes_ = 9
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.row_subsample = row_subsample
        self.min_loss_reduction = min_loss_reduction
        self.column_subsample = column_subsample
        self.step_size = step_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y, sample_weight=None):
        sf = self._array_to_sframe(X, y)
        self.model = gl.boosted_trees_classifier.create(sf, target='target',
                                                        max_iterations=self.max_iterations,
                                                        max_depth=self.max_depth,
                                                        min_child_weight=self.min_child_weight,
                                                        row_subsample=self.row_subsample,
                                                        min_loss_reduction=self.min_loss_reduction,
                                                        column_subsample=self.column_subsample,
                                                        step_size=self.step_size,
                                                        verbose=self.verbose)

        return self

    def predict(self, X):
        preds = self.predict_proba(X)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        sf = self._array_to_sframe(X)
        preds = self.model.predict_topk(sf, output_type='probability', k=self.n_classes_)

        return self._preds_to_array(preds)

    # Private methods
    def _array_to_sframe(self, data, targets=None):
        d = dict()
        for i in xrange(data.shape[1]):
            d['feat_%d' % (i + 1)] = gl.SArray(data[:, i])
        if targets is not None:
            d['target'] = gl.SArray(targets)

        return gl.SFrame(d)

    def _preds_to_array(self, preds):
        p = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
        p['id'] = p['id'].astype(int) + 1
        p = p.sort('id')
        del p['id']
        preds_array = np.array(p.to_dataframe(), dtype=float)

        return preds_array


def run(train_file, test_file, output_file):
    train, labels, test = utils.load_data(train_file, test_file)

    clf = XGBoost(max_iterations=500, max_depth=12, min_child_weight=4.9208250938262745,
                  row_subsample=.9134478530382129, min_loss_reduction=.5132278416508804,
                  column_subsample=.730128689911957, step_size=.1)
    clf.fit(train, labels)
    predictions = clf.predict_proba(test)
    utils.save_prediction(output_file, predictions)

if __name__ == '__main__':
    for i in range(1):
        train_file = os.path.join('data', 'raw', 'train' + str(i) + '.csv')
        test_file = os.path.join('data', 'raw', 'test' + str(i) + '.csv')
        output_file = os.path.join('data', 'prediction', 'xgboost_more' + str(i) + '.csv')
        run(train_file, test_file, output_file)
        #run('data/raw/train.csv', 'data/raw/test.csv', 'data/prediction/xgboost.csv')






