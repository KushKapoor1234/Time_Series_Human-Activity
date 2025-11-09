# tests/test_splits_and_shapes.py
import numpy as np
from src import data as D

def test_loso_indices():
    _,_,subs,_,_,_ = D.get_train_test()
    folds = D.loso_indices(subs)
    # ensure coverage
    idx_union = np.concatenate([te for _,_,te in folds])
    assert np.array_equal(np.sort(idx_union), np.arange(len(subs)))
    # no overlap train/test within fold
    for sid,tr,te in folds:
        assert len(set(tr).intersection(set(te)))==0
