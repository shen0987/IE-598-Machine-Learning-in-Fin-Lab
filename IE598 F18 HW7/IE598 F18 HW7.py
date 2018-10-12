import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
df = pd.read_csv('D:\Downloads\wine.csv')
print (df.head())
print ('Number of Rows of Data =',len(df.index))
print ('Number of Columns of Data =',len(df.columns))
df.describe()

from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['Class'].values
#y=y.reshape(-1,1)

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.model_selection import cross_val_score

for k in [5,10,20,50,75,100]:
    rf = RandomForestClassifier(n_estimators=k,
            random_state=2)
    # Fit rf to the training set    
    rf.fit(X_train,y_train)

    scores_train = cross_val_score(rf, X_train, y_train, cv=10)
    print ("Accuracy of train data: ",scores_train)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train.mean(), scores_train.std() * 2))


from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier(max_depth=2,random_state=0)
params_clf = {
    'n_estimators':[5,10,15,20,25]
}

grid_clf = GridSearchCV(estimator=clf,
                       param_grid=params_clf,
                       scoring='accuracy',
                       cv=10,
                       n_jobs=1)
grid_clf.fit(X_train, y_train)
print (grid_clf.best_params_)
# Extract the best estimator
best_model = grid_clf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Create a pd.Series of features importances
importances = pd.Series(data=best_model.feature_importances_,
                        index= X_train.columns)
# Sort importances
importances_sorted = importances.sort_values()
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh',color='lightgreen')
plt.title('Features Importances')
plt.show()
