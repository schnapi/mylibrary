from sklearn.tree import DecisionTreeClassifier as DTC

X = [[0], [1], [2]]  # 3 simple training examples
Y = [1, 2, 1]  # class labels

dtc = DTC(max_depth=1)

# Case 1: no sample_weight

dtc.fit(X, Y)
print(dtc.tree_.threshold)
# [0.5, -2, -2]
print(dtc.tree_.impurity)
# [0.44444444, 0, 0.5]

# Case 2: with sample_weight
dtc.fit(X, Y, sample_weight=[1, 2, 3])
print(dtc.tree_.threshold)
# [1.5, -2, -2]
print(dtc.tree_.impurity)
# [0.44444444, 0.44444444, 0.]

# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as 
# n_samples / (n_classes * np.bincount(y))

# [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] 