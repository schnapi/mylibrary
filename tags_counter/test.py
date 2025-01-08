
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

path = 'sms.tsv'
features = ['label', 'message']
sms = pd.read_table(path, header=None, names=features)
sms.label.value_counts()
sms['label_num'] = sms.label.map({'ham': 0, 'spam': 1})
X = sms.message
y = sms.label_num
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = CountVectorizer()

# 3. fit
vect.fit(X_train)

# 4. transform training data
X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
