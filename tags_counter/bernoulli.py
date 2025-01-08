
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Create three binary features
X = np.array([[1, 1, 1],
              [0, 1, 0],
              [1, 1, 1],
              [0, 0, 0],
              [1, 0, 1],
              [1, 1, 1],
              [0, 1, 1],
              [1, 1, 1],
              [1, 1, 1],
              [1, 1, 0]])

# Create a binary target vector
y = np.random.randint(2, size=(10, 1)).ravel()

# Create Bernoulli Naive Bayes object with prior probabilities of each class
clf = BernoulliNB()

# Train model
model = clf.fit(X, y)

print(model.predict_proba(X),y)