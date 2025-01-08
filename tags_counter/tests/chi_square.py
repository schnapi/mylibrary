import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.stats import chisquare
from scipy.stats import chi2_contingency

labels = {'restaurant': 0, 'bar': 1, 'cafe': 2, 'hotel': 3}

# y = []  # Create target vector
# text_data = []
# with open('world.class_from_names_with_tags.txt') as fr:
#     for l in fr:
#         temp = l.split(',', 1)
#         tags = [a.strip() for a in temp[1].split(',')]
#         text_data += tags
#         for i in range(len(tags)):
#             y.append(labels[temp[0].strip()])
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(text_data)
# with open('chi_square.pickle', 'wb') as f:
#     pickle.dump([vectorizer, X, y], f)
x = 4
y = (lambda x=x: [x + i for i in range(3)])()

from scipy.stats import chi2 as stat
ar = np.array([[60, 1, 1, 1, 22, 55], [30, 50, 51, 20, 10000, 10], [30, 1, 1, 35, 1, 0]])
c, p, dof, expected = chi2_contingency(ar)
total = ar.sum()
# expected = ar.sum(axis=0) * ar.sum(axis=1) / total
chisq = (ar - expected)**2 / expected
# if p is lower than 0.05% then variables are independet
np.set_printoptions(suppress=True, precision=5)
print('array\n', ar)
print('expected\n', expected)
print('chi2\n', chisq)
print('chi2 sum\n', chisq.sum(axis=0))
# chisq = chisq.sum(axis=0)  # chi2 for each column
print('p\n', stat.sf(chisq, 3))
pass
