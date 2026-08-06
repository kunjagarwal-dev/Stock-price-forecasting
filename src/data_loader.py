import pandas as pd

def data_stock_loader(path="../data/raw/MSFT.csv"):
    df= pd.read_csv(path, index_col="Date", parse_dates=True)
    return df