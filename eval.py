import numpy as np
from baselines import AdaFairClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from fairens import FairAugEnsemble
from fairlearn.postprocessing import ThresholdOptimizer

METRICS = {
    'all': ['acc', 'bacc', 'dp', 'eo'],
    'utility': ['acc', 'bacc'],
    'fairness': ['dp', 'eo'],
}


def evaluate(model, X, y, s, random_state=None, in_percentage=False):
    """
    Evaluate the fairness and performance metrics of a model on the provided dataset.

    Parameters:
        model: The trained predictive model to be evaluated.
        X (numpy array or pandas DataFrame): Input features for evaluation.
        y (numpy array or pandas Series): True labels for evaluation.
        s (numpy array or pandas Series): Sensitive attribute values for evaluation.
        in_percentage (bool, optional): Whether to return the metrics in percentage. Defaults to False.

    Returns:
        dict: A dictionary containing accuracy, demographic parity difference (dp), equalized odds difference (eo),
              accuracy for each group (acc_grp), and positive rate for each group (pos_rate_grp).
    """
    try:
        if isinstance(model, AdaFairClassifier):
            y_pred = model.predict(X)
        elif isinstance(model, FairAugEnsemble):
            y_pred = model.predict(X, sensitive_features=s)
        elif isinstance(model, ThresholdOptimizer):
            y_pred = model.predict(X, sensitive_features=s, random_state=random_state)
        else:
            try:
                y_pred = model.predict(
                    X, sensitive_features=s, random_state=random_state
                )
            except:
                try:
                    y_pred = model.predict(X, sensitive_features=s)
                except:
                    y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f'Failed to predict using the provided model: {e}')

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    dp = demographic_parity_difference(y, y_pred, sensitive_features=s)
    eo = equalized_odds_difference(y, y_pred, sensitive_features=s)
    acc_grp, pos_rate_grp = {}, {}
    g_adv, max_pos_rate = None, 0
    for su in np.unique(s):
        mask = s == su
        acc_grp[su] = accuracy_score(y[mask], y_pred[mask]).round(3)
        pos_rate_grp[su] = np.mean(y_pred[mask]).round(3)
        if pos_rate_grp[su] > max_pos_rate:
            g_adv, max_pos_rate = su, pos_rate_grp[su]
    if in_percentage:
        acc *= 100
        bacc *= 100
        dp *= 100
        eo *= 100
    return {
        'acc': acc,
        'bacc': bacc,
        'dp': dp,
        'eo': eo,
        'acc_grp': acc_grp,
        'pos_rate_grp': pos_rate_grp,
        'g_adv': g_adv,
    }


def evaluate_multi_split(model, data_dict, random_state=None):
    """
    Evaluate a model on multiple datasets.

    Parameters:
        model: The trained predictive model to be evaluated.
        data_dict (dict): A dictionary containing dataset names as keys and tuple of (X, y, s) as values.

    Returns:
        dict: A dictionary containing evaluation results for each dataset in data_dict.
    """
    results = {}
    for data_name, (X, y, s) in data_dict.items():
        results[data_name] = evaluate(model, X, y, s, random_state=random_state)
    return results


def verbose_print(result_dict):
    """
    Print the evaluation results in a formatted and verbose manner.

    Parameters:
        result_dict (dict): A dictionary containing evaluation results for different datasets.
    """
    info = ""
    max_len = max([len(k) for k in result_dict.keys()])
    for data_name, result in result_dict.items():
        info = f"{data_name:<{max_len}s}"
        for metric in METRICS['all']:
            info += f" | {metric}={result[metric]:.3f}"
        info += (
            f" | acc_grp={result['acc_grp']} | pos_rate_grp={result['pos_rate_grp']}"
        )
        print(info)
