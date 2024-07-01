import numpy as np


def all_equal(*t: np.ndarray):
    assert len(t) >= 2
    pair_eqs = [np.equal(t[i], t[i + 1]) for i in range(len(t) - 1)]

    return np.all(pair_eqs)
