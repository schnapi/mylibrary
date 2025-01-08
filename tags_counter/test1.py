
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

text_data = np.array([
    'dejan',
    'dejan1', 'sandi1',
    'sandi'])
labels = [0, 0, 1, 1]
# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Create feature matrix
X = bag_of_words.toarray()
# Create multinomial naive Bayes object with prior probabilities of each class
clf = MultinomialNB(class_prior=[0.5, 0.25])

# Train model
model = clf.fit(X, labels)
# Create new observation

# Predict new observation's class
print(model.predict(count.transform(['dejan'])))


pass
