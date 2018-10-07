import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))


#Impurity
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


accuracylist_train=[]
accuracylist_test=[]

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_train_pred=tree.predict(X_train)
    y_test_pred=tree.predict(X_test)
    print ("Accuracy of in-sample: ",accuracy_score(y_train, y_train_pred))
    print ("Accuracy of out-of-sample:",accuracy_score(y_test, y_test_pred))
    
    accuracylist_train.append(accuracy_score(y_train, y_train_pred))
    accuracylist_test.append(accuracy_score(y_test, y_test_pred))

c={"accuracy of in-sample" : accuracylist_train,"accuracy of out-of-sample" : accuracylist_test}
accuracydata=pd.DataFrame(c)
print ("accuracy dataframe:",accuracydata)
accuracydata.mean()
accuracydata.std()

'''
print ("accuracy list of in-sample:",accuracylist_train)
print ("accuracy list of out-of-sample:",accuracylist_test)
'''

'''
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
'''
#K-fold CV
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    scores_train = cross_val_score(tree, X_train, y_train, cv=10)
    y_train_pred_cv = tree.predict(X_train)
    y_test_pred_cv = tree.predict(X_test)
    print ("=========random_state=",i,"===========")
    print ("Accuracy of in-sample: ",scores_train)
    print ("")
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train.mean(), scores_train.std() * 2))
    print ("Accuracy of out-of-sample:",accuracy_score(y_test, y_test_pred_cv))
    accuracydata_cv=pd.DataFrame(scores_train)
    print ("accuracy dataframe:",accuracydata_cv)
    print ("mean:",accuracydata_cv.mean())
    print ("std",accuracydata_cv.std())

print("My name is Zhanjie Shen")
print("My NetID is: zhanjie2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")