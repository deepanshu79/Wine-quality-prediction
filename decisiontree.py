# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

for i in range(len(y)):
    if y[i]>=7:
        y[i]=1
    else:
        y[i]=0

# Data visualization
men = X[:, 0]
plt.scatter(men, y)
plt.title('fixed acidity vs quality class')
plt.xlabel('fixed acidity')
plt.ylabel('quality class')
plt.show()

glu = X[:, 1]
plt.scatter(glu, y)
plt.title('volatile acidity vs quality class')
plt.xlabel('volatile acidity')
plt.ylabel('quality class')
plt.show()

bld = X[:, 2]
plt.scatter(bld, y)
plt.title('citric acid vs quality class')
plt.xlabel('citric acid')
plt.ylabel('quality class')
plt.show()

skn = X[:, 3]
plt.scatter(skn, y)
plt.title('residual sugar vs quality class')
plt.xlabel('residual sugar')
plt.ylabel('quality class')
plt.show()

ins = X[:, 4]
plt.scatter(ins, y)
plt.title('chlorides vs quality class')
plt.xlabel( 'chlorides')
plt.ylabel('quality class')
plt.show()

bmi = X[:, 5]
plt.scatter(bmi, y)
plt.title('free sulfur dioxide vs quality class')
plt.xlabel('free sulfur dioxide')
plt.ylabel('quality class')
plt.show()

dpf = X[:, 6]
plt.scatter(dpf, y)
plt.title('total sulfur dioxide vs quality class')
plt.xlabel('total sulfur dioxide')
plt.ylabel('quality class')
plt.show()

age = X[:, 7]
plt.scatter(age, y)
plt.title('density vs quality class')
plt.xlabel('density')
plt.ylabel('quality class')
plt.show()

phh = X[:, 8]
plt.scatter(phh, y)
plt.title('pH vs quality class')
plt.xlabel('pH')
plt.ylabel('quality class')
plt.show()

sul = X[:, 9]
plt.scatter(sul, y)
plt.title('sulphates vs quality class')
plt.xlabel('sulphates')
plt.ylabel('quality class')
plt.show()

alc = X[:, 10]
plt.scatter(alc, y)
plt.title('alcohol vs quality class')
plt.xlabel('alcohol')
plt.ylabel('quality class')
plt.show()
      
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# R-Squared and Adjusted R-Squared values
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print("R-square value: "+str(R2))
n = 320
p = 11
AR2 = 1-(1-R2)*(n-1)/(n-p-1)
print("Adjusted R-square value: "+str(AR2))


