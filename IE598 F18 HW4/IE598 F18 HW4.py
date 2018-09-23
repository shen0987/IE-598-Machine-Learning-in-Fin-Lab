import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('D:\Downloads\concrete(1).csv')
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['strength'].values
y=y.reshape(-1,1)

sc_y = StandardScaler()
sc_y.fit(y)
y_std = sc_y.transform(y)
sc_x = StandardScaler()
sc_x.fit(X)
X_std = sc_x.transform(X)
#Split dataset
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=42)

#Regression
slr = LinearRegression()
slr.fit(X_train_std, y_train)
y_train_pred = slr.predict(X_train_std)
y_test_pred = slr.predict(X_test_std)


#describe the regression
print ('Regression coefficients:',slr.coef_)
print ('Regression intercept:',slr.intercept_)

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


#Ridge Regression
from sklearn.linear_model import Ridge
for a in [0.00001,0.0001,0.001,0.01, 0.1, 0.5, 1, 10]:
     ridge = Ridge(alpha=a)
     ridge.fit(X_train_std, y_train)
     y_train_pred = ridge.predict(X_train_std)
     y_test_pred = ridge.predict(X_test_std)
     print('===============','Alpha=',a,'===============')
     #print('Training accuracy:', ridge.score(X_train_std, y_train),'Test accuracy:', ridge.score(X_test_std, y_test))

     plt.scatter(y_train_pred,  y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
     plt.scatter(y_test_pred,  y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
     plt.xlabel('Predicted values')
     plt.ylabel('Residuals')
     plt.legend(loc='upper left')
     plt.hlines(y=0, xmin=-2, xmax=2, color='black', lw=2)
     plt.xlim([-2, 2])
     plt.show()
     print ('Coefficients:',ridge.coef_)
     print ('Intercept:',ridge.intercept_)
     print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
     print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas=[0.00001,0.0001,0.001,0.01, 0.1, 0.5, 1, 10])
ridgecv.fit(X_std, y_std)
print ('-----------The alpha gives the best performing model:',ridgecv.alpha_)


#LASSO Regression
from sklearn.linear_model import Lasso

for l in [0.00001,0.0001,0.001,0.01, 0.1, 0.5, 1, 10]:
     lasso = Lasso(alpha=l)
     lasso.fit(X_train_std, y_train)
     y_train_pred = lasso.predict(X_train_std)
     y_test_pred = lasso.predict(X_test_std)
     y_train_pred=y_train_pred.reshape(-1,1)
     y_test_pred=y_test_pred.reshape(-1,1)
     print('===============','Alpha=',l,'===============')
     #print('Training accuracy:', ridge.score(X_train_std, y_train),'Test accuracy:', ridge.score(X_test_std, y_test))

     plt.scatter(y_train_pred,  y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
     plt.scatter(y_test_pred,  y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
     plt.xlabel('Predicted values')
     plt.ylabel('Residuals')
     plt.legend(loc='upper left')
     plt.hlines(y=0, xmin=-2, xmax=2, color='black', lw=2)
     plt.xlim([-2, 2])
     plt.show()
     print ('Coefficients:',lasso.coef_)
     print ('Intercept:',lasso.intercept_)
     print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
     print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

from sklearn.linear_model import LassoCV
lassocv = LassoCV(alphas=[0.00001,0.0001,0.001,0.01, 0.1, 0.5, 1, 10])
lassocv.fit(X_std, y_std)
print ('-----------The alpha gives the best performing model:',lassocv.alpha_)


#Elastic Net
from sklearn.linear_model import ElasticNet

for e in [0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]:
     elanet = ElasticNet(alpha=1.0, l1_ratio=e)
     elanet.fit(X_train_std, y_train)
     y_train_pred = elanet.predict(X_train_std)
     y_test_pred = elanet.predict(X_test_std)
     y_train_pred=y_train_pred.reshape(-1,1)
     y_test_pred=y_test_pred.reshape(-1,1)
     print('===============','l1_ratio=',e,'===============')
     #print('Training accuracy:', ridge.score(X_train_std, y_train),'Test accuracy:', ridge.score(X_test_std, y_test))

     plt.scatter(y_train_pred,  y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
     plt.scatter(y_test_pred,  y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
     plt.xlabel('Predicted values')
     plt.ylabel('Residuals')
     plt.legend(loc='upper left')
     plt.hlines(y=0, xmin=-2, xmax=2, color='black', lw=2)
     plt.xlim([-1.5, 1.5])
     plt.show()
     print ('Coefficients:',elanet.coef_)
     print ('Intercept:',elanet.intercept_)
     print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
     print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

from sklearn.linear_model import ElasticNetCV
elanetcv = ElasticNetCV(alphas=[1],l1_ratio=[0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0])
elanetcv.fit(X_std, y_std)
print ('-----------The l1_ratio gives the best performing model:',elanetcv.l1_ratio_)

print("My name is Zhanjie Shen")
print("My NetID is: zhanjie2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")