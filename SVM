from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import chardet
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#clf = SVC(kernel = 'linear')
#with open(r'C:\Users\BENJI\Documents\IRIS.csv', 'rb') as f:
 #   result = chardet.detect(f.readline())
#X =pd.read_csv(r"C:\Users\BENJI\Documents\IRIS.csv", encoding = result['encoding'])
#a=np.array(X)
iris=datasets.load_iris()
X=iris.data[:100,:2]
Y=iris.target[:100]
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
svc=LinearSVC(C=1.0)
model=svc.fit(X_std,Y)
color=['black' if c==0 else 'lightgrey' for c in Y]
plt.scatter(X_std[:,0],X_std[:,1], c=color)
w=svc.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(-2.5,2.5)
yy=a*xx - (svc.intercept_[0])/w[1]
plt.plot(xx,yy)
plt.show()
#x=a[:,0:4]
#y=iris.target
#X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.5)
#clf.fit(X_train,Y_train)
#W=clf.predict(X_test)
#e=sklearn.metrics.accuracy_score(Y_test,W,normalize=True,sample_weight=None)
#plt.plot(Y_train,X_train)
#plt.show()
#print (e)
#X, Y = iris(n_samples=150, centers=2, random_state=0, cluster_std=0.40)
#xfit = np.linspace(-1, 3.5)
#print(xfit)
#plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
#for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#    yfit = m * xfit + b
#    plt.plot(xfit, yfit, '-k')
#    plt.fill_between(xfit, yfit - d, edgecolor='none', color='#AAAAAA', alpha= 0.4)
#plt.xlim(-1, 3.5); 
#plt.ylim(0, 5);   
#plt.show()
