def train_test_split_series(series, train_ratio=0.9):
    """Split a time series into train/test preserving order."""
    train_size = int(len(series) * train_ratio)
    train = series[:train_size]
    test = series[train_size:]
    return train, test