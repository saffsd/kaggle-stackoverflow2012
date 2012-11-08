"""
Marco Lui, October 2012
"""
import argparse
import numpy as np

# from http://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/2644/multi-class-log-loss-function
import numpy as np

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota



eps=1e-15

if __name__ == "__main__": 
  #multiclass_log_loss(np.array([0,1,2]),np.array([[1,0,0],[0,1,0],[0,0,1]]))
  # TODO check that labels match
  parser = argparse.ArgumentParser(description="Evaluate an output using multiclass log loss")
  parser.add_argument("true")
  parser.add_argument("pred")
  args = parser.parse_args()

  with open(args.true) as true_f:
    data_true = np.array([ int(row[0])-1 for row in true_f ], dtype=int)
  data_pred = np.genfromtxt(args.pred, delimiter=',')

  mll = multiclass_log_loss(data_true, data_pred)

  print "{0}".format(mll)

