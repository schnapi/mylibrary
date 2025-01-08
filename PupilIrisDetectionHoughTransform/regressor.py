# Decision Tree Regression
import pandas
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
columns = ['isarcus', 'inside_iris_l',  'edge_inside_iris_l',
           'between_edges_l', 'edge_out_iris_l', 'out_iris_l',
           'inside_iris_r',  'edge_inside_iris_r', 'between_edges_r',
           'edge_out_iris_r', 'out_iris_r',
           'std_inside_iris_l',  'std_edge_inside_iris_l',
           'std_between_edges_l', 'std_edge_out_iris_l', 'std_out_iris_l',
           'std_inside_iris_r',  'std_edge_inside_iris_r', 'std_between_edges_r',
           'std_edge_out_iris_r', 'std_out_iris_r',
           'thresh', 'minstd', 'minmean']
dataframe = pandas.read_csv('arcus.csv', names=columns)
array = dataframe.values
print(dataframe.head())
X = array[:, 1:]
Y = array[:, 0]

# x_index = columns.index('std_out_iris_r')
x_index = columns.index('std_edge_out_iris_l')
y_index = columns.index('edge_out_iris_l')
x_index = columns.index('between_edges_r')
y_index = columns.index('std_between_edges_r')
# x_index = columns.index('between_edges_l')
# y_index = columns.index('std_between_edges_l')
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: ['healty', 'arcus s'][int(i)])

plt.scatter(array[:, x_index], array[:, y_index],  c=Y, cmap=plt.cm.get_cmap('RdYlBu', 2))
plt.colorbar(ticks=[0, 1], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(columns[x_index])
plt.ylabel(columns[y_index])
plt.show()





selection = [1, 2, 4]
selection = [2, 3, 4]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results)
model.fit(X, Y)

import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(model, filled=True, out_file=None, feature_names=list(np.array(columns)[1:]))
graph = graphviz.Source(dot_data)
graph.render("iris1")
