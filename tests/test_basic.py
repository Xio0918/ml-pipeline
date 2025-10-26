import pandas as pd
from src.preprocess import load_data, preprocess

def test_load_and_preprocess():
    df = load_data("data/winequality-red.csv")
    assert isinstance(df, pd.DataFrame)
    X, y = preprocess(df)
    assert X.shape[0] == y.shape[0]
