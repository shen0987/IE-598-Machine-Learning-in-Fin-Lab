#Fit a KNN and decision tree classifier to the Iris dataset

import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print( X_train.shape, y_train.shape)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = X_train_std[:, 0][y_train == i]
    ys = X_train_std[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

#Decision tree
import matplotlib.pyplot as plt
import numpy as np
def gini(p):return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],  ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],['-', '-', '--', '-.'],['black', 'lightgray','red', 'green', 'cyan']):line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
tree.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined, clf=tree)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree,filled=True, rounded=True,class_names=['Setosa', 'Versicolor','Virginica'], feature_names=['petal length', 'petal width'],out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png')


#KNN and select the best K
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, p=2,metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred=knn.predict(X_test_std)
    scores.append(accuracy_score(y_test,y_pred))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
print (scores)
k=[]
Maxi=max(scores)
print ("K with the best accuracy result is/are: ")
for i in range(len(scores)):
    if scores[i]==Maxi:
        k.append(i+1)
        print ("========== k is",i+1," ==========")
        knn = KNeighborsClassifier(n_neighbors=i+1, p=2,metric='minkowski')
        knn.fit(X_train_std, y_train)
        y_pred=knn.predict(X_test_std)
        scores.append(accuracy_score(y_test,y_pred))
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))
        plot_decision_regions(X_combined_std, y_combined, clf=knn)
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.show()
print ("k=", k," Best accuracy is:", max(scores))

print("My name is Zhanjie Shen")
print("My NetID is zhanjie2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

