import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

from my_toolbox.preprocessing import quick_ohe_binary

def test_quick_ohe_binary():
    df = pd.DataFrame({'test':['low', 'high', 'high']})
    assert quick_ohe_binary(df, 'test').test.sum() == 1