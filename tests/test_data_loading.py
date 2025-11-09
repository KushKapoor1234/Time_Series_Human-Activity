# tests/test_data_loading.py
import numpy as np
from src import data as D

def test_shapes():
    Xtr,ytr,str_,Xte,yte,ste_ = D.get_train_test()
    assert Xtr.ndim==3 and Xte.ndim==3
    n_steps=128; n_feat=9
    assert Xtr.shape[1]==n_steps and Xtr.shape[2]==n_feat
    assert Xte.shape[1]==n_steps and Xte.shape[2]==n_feat
    assert len(Xtr)==len(ytr)==len(str_)
    assert len(Xte)==len(yte)==len(ste_)
    assert (ytr.min()>=0) and (ytr.max()<=5)
