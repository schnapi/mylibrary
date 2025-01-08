from sklearn.model_selection import ShuffleSplit


X = range(100)
rs = ShuffleSplit(len(X), 0.1, .3)
for train_idx, test_idx in rs.split(X):
    print(train_idx, test_idx)