import dill
import numpy as np
from sklearn.linear_model import LogisticRegression

total_samples = 500
assert total_samples % 2 == 0
num_advantaged = total_samples // 2
num_disadvantaged = total_samples // 2

def gaussian_noise(n_feat):
    return np.random.randn(n_feat)

def noisy_feature(feature, bias, variance):
    variance = gaussian_noise(len(feature)) * variance
    return (feature + bias) + variance

def noisy_proxy(a, bias, variance):
    advantaged_cond = (a==1)
    disadvantaged_cond = (a==0)
    advantaged_noise = sorted(np.clip(gaussian_noise(np.sum(advantaged_cond)*2), a_min=0, a_max=None))[-np.sum(advantaged_cond):]
    disadvantaged_noise = sorted(np.clip(gaussian_noise(np.sum(disadvantaged_cond)*2), a_min=None, a_max=0))[:np.sum(disadvantaged_cond)]
    feature = a.copy().astype('float64')
    feature[advantaged_cond] = advantaged_noise
    feature[disadvantaged_cond] = disadvantaged_noise + max(max(advantaged_noise), -min(disadvantaged_noise)) + 0.5
    feature /= np.max(feature) - np.min(feature)
    return feature

def noisy_oracle(y, bias, variance):
    return noisy_feature(y, bias, variance)

def sensitive_feature_biased(a, y, advantaged_bias, disadvantaged_bias, advantaged_variance, disadvantaged_variance):
    '''
    Similar to feature: Performance Review
    Amount of unfairness: difference in bias or variance between the two groups.
    '''
    feature = np.zeros_like(a).astype('float64')
    bias_variance = {
        0: (disadvantaged_bias, disadvantaged_variance),
        1: (advantaged_bias, advantaged_variance),
    }
    for av in range(2):
        a_cond = (a==av)
        a_feature = gaussian_noise(len(y[a_cond]))
        bias, variance = bias_variance[av]
        a_feature = noisy_feature(a_feature, bias, variance)
        sorted_a_feature = sorted(a_feature, reverse=True)
        a_y1 = a_cond&(y==1)
        a_y0 = a_cond&(y==0)
        n_a_y1 = sum(a_y1)
        feature[a_y1] = sorted_a_feature[:n_a_y1]
        feature[a_y0] = sorted_a_feature[n_a_y1:]
    return feature

def generate_A_Y(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged):
    num_samples = num_advantaged + num_disadvantaged
    Y = np.random.randint(2, size=num_samples)
    A = np.array([1]*num_advantaged + [0]*num_disadvantaged)
    ''' 
    group     2a + y
    a=0, y=0    0
    a=0, y=1    1
    a=1, y=0    2
    a=1, y=1    3
    '''
    g = np.array([2*a + y for a, y in zip(A, Y)])
    g_labels = {
        0: "(a=0, y=0)",
        1: "(a=0, y=1)",
        2: "(a=1, y=0)",
        3: "(a=1, y=1)",
    }
    return A, Y, g, g_labels

def dataset_1(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged):
    '''
    x1: noisy oracle
    x2: noisy oracle
    '''
    A, Y, g, g_labels = generate_A_Y(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged)
    x1 = noisy_oracle(Y, 3, 0.25)
    x2 = noisy_oracle(Y, 2, 0.3)
    X = np.vstack([x1, x2]).T

    # Is linearly separable
    # TODO: Calculate based on noisy oracle values
    nA, nY, ng, ng_labels = generate_A_Y(num_advantaged=10000, num_disadvantaged=10000)
    nx1 = noisy_oracle(nY, 3, 0.25)
    nx2 = noisy_oracle(nY, 2, 0.3)
    nX = np.vstack([nx1, nx2]).T

    clf = LogisticRegression().fit(nX, nY)
    def opt_fair_clf(ax, x_min, x_max, y_min, y_max):
        # Retrieve the model parameters.
        b = clf.intercept_[0]
        w1, w2 = clf.coef_.T
        # Calculate the intercept and gradient of the decision boundary.
        c = -b/w2
        m = -w1/w2

        xd = np.array([x_min, x_max])
        yd = m*xd + c
        ax.plot(xd, yd, color='red', ls='--', label='Optimal Fair Boundary')
        return ax

    return X, Y, A, opt_fair_clf

def dataset_2(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged):
    '''
    x1: noisy oracle
    x2: random noise
    '''
    A, Y, g, g_labels = generate_A_Y(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged)
    bias = 3
    x1 = noisy_oracle(Y, bias, 0.25)
    x2 = gaussian_noise(len(Y))
    X = np.vstack([x1, x2]).T
    
    # Is the line in between y=0 and y=1 (with bias term) in the noisy oracle
    def opt_fair_clf(ax, x_min, x_max, y_min, y_max):
        x_line = bias + ((min(Y) + max(Y)) / 2)
        xd = np.array([x_line, x_line])
        yd = np.array([y_min, y_max])
        ax.plot(xd, yd, color='red', ls='--', label='Optimal Fair Boundary')
        return ax

    return X, Y, A, opt_fair_clf

def dataset_3(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged):
    '''
    x1: noisy oracle
    x2: noisy proxy
    '''
    A, Y, g, g_labels = generate_A_Y(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged)
    bias = 3
    x1 = noisy_oracle(Y, bias, 0.25)
    x2 = noisy_proxy(A, bias, 0.25)
    X = np.vstack([x1, x2]).T
    
    # Is the line in between y=0 and y=1 (with bias term) in the noisy oracle
    def opt_fair_clf(ax, x_min, x_max, y_min, y_max):
        x_line = bias + ((min(Y) + max(Y)) / 2)
        xd = np.array([x_line, x_line])
        yd = np.array([y_min, y_max])
        ax.plot(xd, yd, color='red', ls='--', label='Optimal Fair Boundary')
        return ax

    return X, Y, A, opt_fair_clf

def dataset_4(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged, plt_hist=False):
    '''
    x1: noisy oracle
    x2: sensitive bias feature
    '''
    A, Y, g, g_labels = generate_A_Y(num_advantaged=num_advantaged, num_disadvantaged=num_disadvantaged)
    x2 = sensitive_feature_biased(A, Y, 7, 6.5, 0.2, 0.25)
    bias = 3
    x1 = noisy_oracle(Y, bias, 0.25)
    x2 -= np.min(x2)
    x2 /= np.max(x2) - np.min(x2)
    x2 *= 10
    if plt_hist:
        for a in range(2):
            for y in range(2):
                plt.hist(x2[(A==a)&(Y==y)], label=f"A={a}, Y={y}")
    X = np.vstack([x1, x2]).T
    
    # Is the line in between y=0 and y=1 (with bias term) in the noisy oracle
    def opt_fair_clf(ax, x_min, x_max, y_min, y_max):
        x_line = bias + ((min(Y) + max(Y)) / 2)
        xd = np.array([x_line, x_line])
        yd = np.array([y_min, y_max])
        ax.plot(xd, yd, color='red', ls='--', label='Optimal Fair Boundary')
        return ax

    return X, Y, A, opt_fair_clf

def subsample(X, Y, A, a0y0, a0y1, a1y0, a1y1):
    # Make list of all indices
    idxs = np.arange(len(Y))
    # Shuffle indices
    np.random.shuffle(idxs)
    # Shuffle X, Y, A according to indice
    X = X[idxs]
    Y = Y[idxs]
    A = A[idxs]
    idxs = np.arange(len(Y))
    # For each (a, y) group, keep only the first a{a}y{y} items.
    ay_size = {
        (0, 0): a0y0,
        (0, 1): a0y1,
        (1, 0): a1y0,
        (1, 1): a1y1,
    }
    subsampled_idxs = []
    for a in range(2):
        for y in range(2):
            group_idxs = idxs[(A==a)&(Y==y)]
            subsampled_idxs.extend(group_idxs[:ay_size[(a, y)]])
    return X[subsampled_idxs], Y[subsampled_idxs], A[subsampled_idxs]

def biased_misclassify_y(y, a, n_samples):
    assert n_samples % 2 == 0
    n_samples = n_samples // 2
    misclassified = y.copy()
    idxs = np.arange(len(y))

    misclassify_to_y0 = (a==0)&(y==1)
    assert np.sum(misclassify_to_y0) >= n_samples
    misclassify_to_y0_idxs = np.random.choice(idxs[misclassify_to_y0], size=n_samples, replace=False)
    misclassified[misclassify_to_y0_idxs] = 0

    misclassify_to_y1 = (a==1)&(y==0)
    assert np.sum(misclassify_to_y1) >= n_samples
    misclassify_to_y1_idxs = np.random.choice(idxs[misclassify_to_y1], size=n_samples, replace=False)
    misclassified[misclassify_to_y1_idxs] = 1

    return misclassified

def biased_misclassify_0(samples, n_samples):
    '''
    Only misclassify sample=0 to sample=1.
    '''
    samples = samples.copy()
    samples0_cond = samples==0
    assert np.sum(samples0_cond) >= n_samples
    idxs = np.random.choice(np.arange(len(samples))[samples0_cond], size=n_samples, replace=False)
    # Flip label
    samples[idxs] = 1
    return samples

def apply_unfair_disadvantaged_variance(param, dataset_fn):
    X, Y, A, opt_fair_clf = dataset_fn()
    cond = (A==0)
    X[:, 0][cond] += np.random.randn(X[cond].shape[0]) * param[0]
    X[:, 1][cond] += np.random.randn(X[cond].shape[0]) * param[1]
    return X, Y, A, opt_fair_clf

def apply_unfair_disadvantaged_group_y0_proportion(param, dataset_fn):
    X, Y, A, opt_fair_clf = dataset_fn()
    y0_size = np.sum(Y==0)
    a0y0_size = int(y0_size*param)
    a1y0_size = y0_size-a0y0_size
    y1_size = np.sum(Y==1)
    a1y1_size = int(y1_size*param)
    a0y1_size = y1_size - a1y1_size
    X, Y, A, opt_fair_clf = dataset_fn(num_advantaged=int(len(Y)*2), num_disadvantaged=int(len(Y)*2))
    X, Y, A = subsample(X, Y, A, a0y0_size, a0y1_size, a1y0_size, a1y1_size)
    return X, Y, A, opt_fair_clf

def apply_unfair_y_bias(param, dataset_fn):
    X, Y, A, opt_fair_clf = dataset_fn()
    n_samples = int(min(np.sum((A==1)&(Y==0)), np.sum((A==0)&(Y==1)))*param)
    if n_samples % 2 != 0:
        n_samples -= 1
    if n_samples == 0:
        n_samples = 2
    Y = biased_misclassify_y(Y, A, n_samples)
    return X, Y, A, opt_fair_clf

def apply_unfair_a_noise(param, dataset_fn):
    X, Y, A, opt_fair_clf = dataset_fn()
    n_samples = int(np.sum(A==0)*param)
    if n_samples == np.sum(A==0):
        n_samples -= 1
    if n_samples % 2 != 0:
        n_samples -= 1
    A = biased_misclassify_0(A, n_samples)
    return X, Y, A, opt_fair_clf

synthetic_datasets = {
    "fair": {
        # Y + N(3, sqrt(0.25)), Y + N(2, sqrt(0.3))
        "Noisy Oracle / Noisy Oracle": dataset_1(),
        # Y + N(3, sqrt(0.25)), N(0, 1)
        "Noisy Oracle / Gaussian Noise": dataset_2(),
        # Y + N(3, sqrt(0.25)), A + N(3, sqrt(0.25))
        "Noisy Oracle / Noisy Proxy": dataset_3(),
    },
    "biased": {
        # Adjust variance of disadvataged group (A=0)
        # x1 + N(0, 1)*0.4
        # x2 + N(0, 1)*0.2
        "(Noisy Oracle / Noisy Oracle) + Less Predictive Power on A=0": apply_unfair_disadvantaged_variance((0.4, 0.2), dataset_1),
        # Increase proportion of disadvataged group (A=0) in the negative outcome
        # and decrease it in the positive outcome.
        # Y=0 A=0 0.668
        # Y=1 A=1 0.669
        "(Noisy Oracle / Noisy Proxy) + Correlate A & Y": apply_unfair_disadvantaged_group_y0_proportion(0.668, dataset_3),
        "Noisy Oracle / Biased Feature": dataset_4(),
        # Adjust variance of disadvataged group (A=0)
        # x1 + N(0, 1)*0.2
        # x2 + N(0, 1)*0.4
        "(Noisy Oracle / Biased Feature) + Less Predictive Power on A=0": apply_unfair_disadvantaged_variance((0.2, 0.4), dataset_4),
        # Misclassify A=0, Y=1 --> A=0, Y=0
        # Misclassify A=1, Y=0 --> A=1, Y=1
        "Noisy Oracle / Biased Feature": apply_unfair_y_bias(0.166, dataset_4),
        # Misclassify A=0 to A=1
        "Noisy Oracle / Biased Feature": apply_unfair_a_noise(0.332, dataset_4),
    },
}

dill.dump(synthetic_datasets, open('synthetic_datasets.pkl', 'wb'))