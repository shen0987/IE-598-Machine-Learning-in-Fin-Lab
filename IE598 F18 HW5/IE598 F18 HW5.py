import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('D:\Downloads\wine.csv')

print (df.head())
print ('Number of Rows of Data =',len(df.index))
print ('Number of Columns of Data =',len(df.columns))
df.describe()

#4Q
print (df.quantile([0,0.25,0.5,0.75,1]))

#QQ Plot
import pylab
import scipy.stats as stats
for k in range(df.shape[1]):
    stats.probplot(df.iloc[:,k], dist="norm", plot=pylab)
    pylab.show()


#Box Plot
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
sns.boxplot(data=df)
 #removing is okay but renormalizing the 

#plot correlation seperately
sns.pairplot(df)
plt.xlabel("Attribute")
plt.ylabel(("Attributes"))
plt.tight_layout()
plt.show()

#plot heatmap
import numpy as np
#cm = np.corrcoef(df[cols].values.T)
cm = np.corrcoef(df.values.T)
sns.set(font_scale=1.5)
#hm = sns.heatmap(cm, cbar=True,annot=True, square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
hm = sns.heatmap(cm, cbar=True,annot=True, square=True,fmt='.2f',annot_kws={'size': 7},yticklabels=df.columns,xticklabels=df.columns)
plt.show()

#fit regression
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['Class'].values
#y=y.reshape(-1,1)

#Standardize
'''
sc_y = StandardScaler()
sc_y.fit(y)
y_std = sc_y.transform(y)
'''
sc_x = StandardScaler()
sc_x.fit(X)
X_std = sc_x.transform(X)

#Split dataset
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

#Regression
from sklearn.linear_model import LogisticRegression
slr = LogisticRegression(C=100.0, random_state=1)
slr.fit(X_train_std, y_train)
y_train_pred = slr.predict(X_train_std)
y_test_pred = slr.predict(X_test_std)


#describe the regression
print ('Regression coefficients:',slr.coef_)
print ('Regression intercept:',slr.intercept_)

'''
#Residual plot
plt.scatter(y_train_pred,  y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-2, xmax=2, color='black', lw=2)
plt.xlim([-2, 2])
plt.show()

#Calculate MSE
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

#Calculate R-square
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
'''

#Calculate CV Accuracy Score
from sklearn.metrics import accuracy_score
print ("===========Logistic=============")
from sklearn.model_selection import cross_val_score
scores_train_log = cross_val_score(slr, X_train_std, y_train, cv=5)
scores_test_log = cross_val_score(slr, X_test_std, y_test, cv=5)
print ("Accuracy of train data: ",scores_train_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_log.mean(), scores_train_log.std() * 2))
print ("Accuracy of test data:",scores_test_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_log.mean(), scores_test_log.std() * 2))
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred))


#SVM regression
print ("===========SVM=============")
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
scores_train_svm = cross_val_score(svm, X_train_std, y_train, cv=5)
scores_test_svm = cross_val_score(svm, X_test_std, y_test, cv=5)
y_train_pred_svm = svm.predict(X_train_std)
y_test_pred_svm = svm.predict(X_test_std)
print ("Accuracy of train data: ",scores_train_svm)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_svm.mean(), scores_train_svm.std() * 2))
print ("Accuracy of test data:",scores_test_svm)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_svm.mean(), scores_test_svm.std() * 2))
print("Accuracy score of train data: %0.2f"%accuracy_score(y_train, y_train_pred_svm))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_svm))

#PCA Logistic
print ("===========PCA Logistic=============")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lr = LogisticRegression(C=100.0, random_state=1)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
scores_train_pca_log = cross_val_score(lr, X_train_pca, y_train, cv=5)
scores_test_pca_log = cross_val_score(lr, X_test_pca, y_test, cv=5)
y_train_pred_pcal = lr.predict(X_train_pca)
y_test_pred_pcal = lr.predict(X_test_pca)
print ("Accuracy of train data: ",scores_train_pca_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_pca_log.mean(), scores_train_pca_log.std() * 2))
print ("Accuracy of test data:",scores_test_pca_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_pca_log.mean(), scores_test_pca_log.std() * 2))
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_pcal))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_pcal))

#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_pca, y_train, clf=lr)

#PCA SVM
print ("===========PCA SVM=============")
pcasvm = SVC(kernel='linear', C=1.0, random_state=1)
pcasvm.fit(X_train_pca, y_train)
scores_train_pca_SVM = cross_val_score(pcasvm, X_train_pca, y_train, cv=5)
scores_test_pca_SVM = cross_val_score(pcasvm, X_test_pca, y_test, cv=5)
y_train_pred_pcas = pcasvm.predict(X_train_pca)
y_test_pred_pcas = pcasvm.predict(X_test_pca)
print ("Accuracy of train data: ",scores_train_pca_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_pca_SVM.mean(), scores_train_pca_SVM.std() * 2))
print ("Accuracy of test data:",scores_test_pca_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_pca_SVM.mean(), scores_test_pca_SVM.std() * 2))
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_pcas))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_pcas))


#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_pca, y_train, clf=pcasvm)

#LDA Logistic
print ("===========LDA Logistic=============")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda=lda.transform(X_test_std)
lrlda = LogisticRegression(C=100.0, random_state=1)
lrlda.fit(X_train_lda, y_train)
scores_train_lda_log = cross_val_score(lrlda, X_train_lda, y_train, cv=5)
scores_test_lda_log = cross_val_score(lrlda, X_test_lda, y_test, cv=5)
print ("Accuracy of train data: ",scores_train_lda_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_lda_log.mean(), scores_train_lda_log.std() * 2))
print ("Accuracy of test data:",scores_test_lda_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_lda_log.mean(), scores_test_lda_log.std() * 2))
y_train_pred_ldal = lrlda.predict(X_train_lda)
y_test_pred_ldal = lrlda.predict(X_test_lda)
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_ldal))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_ldal))


#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_lda, y_train, clf=lrlda)

#LDA SVM
print ("===========LDA SVM=============")
ldasvm = SVC(kernel='linear', C=1.0, random_state=1)
ldasvm.fit(X_train_lda, y_train)
scores_train_lda_SVM = cross_val_score(ldasvm, X_train_lda, y_train, cv=5)
scores_test_lda_SVM = cross_val_score(ldasvm, X_test_lda, y_test, cv=5)
print ("Accuracy of train data: ",scores_train_lda_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_lda_SVM.mean(), scores_train_lda_SVM.std() * 2))
print ("Accuracy of test data:",scores_test_lda_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_lda_SVM.mean(), scores_test_lda_SVM.std() * 2))
y_train_pred_ldas = ldasvm.predict(X_train_lda)
y_test_pred_ldas = ldasvm.predict(X_test_lda)
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_ldas))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_ldas))



#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_lda, y_train, clf=ldasvm)

#kPCA Logistic
print ("===========KPCA Logistic=============")
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_skernpca = scikit_kpca.fit_transform(X_train_std)
X_test_skernpca = scikit_kpca.transform(X_test_std)
lrk = LogisticRegression(C=100.0, random_state=1)
lrk.fit(X_train_skernpca, y_train)
scores_train_k_log = cross_val_score(lrk, X_train_skernpca, y_train, cv=5)
scores_test_k_log = cross_val_score(lrk, X_test_skernpca, y_test, cv=5)
print ("Accuracy of train data: ",scores_train_k_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_k_log.mean(), scores_train_k_log.std() * 2))
print ("Accuracy of test data:",scores_test_k_log)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_k_log.mean(), scores_test_k_log.std() * 2))
y_train_pred_kl = lrk.predict(X_train_skernpca)
y_test_pred_kl = lrk.predict(X_test_skernpca)
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_kl))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_kl))


#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_skernpca, y_train, clf=lrk)

#kPCA SVM
print ("===========KPCA SVM=============")
ksvm = SVC(kernel='linear', C=1.0, random_state=1)
ksvm.fit(X_train_skernpca, y_train)
scores_train_k_SVM = cross_val_score(ksvm, X_train_skernpca, y_train, cv=5)
scores_test_k_SVM = cross_val_score(ksvm, X_test_skernpca, y_test, cv=5)
print ("Accuracy of train data: ",scores_train_k_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_train_k_SVM.mean(), scores_train_k_SVM.std() * 2))
print ("Accuracy of test data:",scores_test_k_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test_k_SVM.mean(), scores_test_k_SVM.std() * 2))
y_train_pred_ks = ksvm.predict(X_train_skernpca)
y_test_pred_ks = ksvm.predict(X_test_skernpca)
print("Accuracy score: %0.2f"%accuracy_score(y_train, y_train_pred_ks))
print("Accuracy score of test data: %0.2f"%accuracy_score(y_test, y_test_pred_ks))


#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X_train_skernpca, y_train, clf=ksvm)