\#!/usr/bin/env python

# coding: utf-8

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make\_column\_transformer
from sklearn.model\_selection import train\_test\_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1\_score, jaccard\_score, log\_loss, confusion\_matrix
from matplotlib.colors import ListedColormap
import types
import ibm\_boto3
from botocore.client import Config

# IBM Cloud Object Storage client setup

client = ibm\_boto3.client(
service\_name='s3',
ibm\_api\_key\_id=APIKEY,
ibm\_auth\_endpoint="[https://iam.cloud.ibm.com/oidc/token](https://iam.cloud.ibm.com/oidc/token)",
config=Config(signature\_version='oauth'),
endpoint\_url='[https://s3-api.us-geo.objectstorage.service.networklayer.com](https://s3-api.us-geo.objectstorage.service.networklayer.com)'
)

# Load dataset

body = client.get\_object(Bucket='finalprojectcoursera-donotdelete-pr-nabcaau098l0pp', Key='loan\_train.csv')\['Body']
if not hasattr(body, "**iter**"):
body.**iter** = types.MethodType(lambda self: iter(()), body)

dataset = pd.read\_csv(body)

# Split dataset

X = dataset.iloc\[:, 0:12].values
y = dataset.iloc\[:, 12].values

# Impute missing values with most frequent strategy

cols\_to\_impute = \[(1, 4), (5, 6), (8, 11)]
for start, end in cols\_to\_impute:
imputer = SimpleImputer(missing\_values=np.nan, strategy='most\_frequent')
X\[:, start\:end] = imputer.fit\_transform(X\[:, start\:end])

# One-hot encode categorical columns

ct\_X = make\_column\_transformer(
(OneHotEncoder(), \[0, 1, 2, 3, 4, 5, 11]), remainder='passthrough')
X = ct\_X.fit\_transform(X).toarray()

# Encode target variable

le = LabelEncoder()
y = le.fit\_transform(y)

# Train-test split

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=0)

# Feature scaling

sc = StandardScaler()
X\_train = sc.fit\_transform(X\_train)
X\_test = sc.transform(X\_test)

# PCA for feature reduction

pca = PCA(n\_components=2)
X\_train\_pca = pca.fit\_transform(X\_train)
X\_test\_pca = pca.transform(X\_test)
explained\_variance = pca.explained\_variance\_ratio\_

# KNN classification and evaluation

k\_range = range(1, 10)
f1\_scores, js\_scores, log\_losses = \[], \[], \[]
for k in k\_range:
knn = KNeighborsClassifier(n\_neighbors=k)
knn.fit(X\_train\_pca, y\_train)
y\_pred = knn.predict(X\_test\_pca)
f1\_scores.append(f1\_score(y\_test, y\_pred))
js\_scores.append(jaccard\_score(y\_test, y\_pred))
log\_losses.append(log\_loss(y\_test, y\_pred))

# Plot KNN performance metrics

plt.style.use("seaborn")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = \[(f1\_scores, 'F1 Score'), (js\_scores, 'Jaccard'), (log\_losses, 'Log Loss')]
colors = \['blue', 'red', 'orange']

for i, (metric, title) in enumerate(metrics):
axes\[i].plot(k\_range, metric, color=colors\[i])
axes\[i].set\_title(f'KNN {title}')
axes\[i].set\_xlabel('Value of K')
axes\[i].set\_ylabel('Score')
axes\[i].set\_xticks(list(k\_range))

plt.tight\_layout()
plt.show()

# Final KNN model

knn\_final = KNeighborsClassifier(n\_neighbors=9, metric='minkowski', p=2)
knn\_final.fit(X\_train\_pca, y\_train)
y\_pred\_knn = knn\_final.predict(X\_test\_pca)

# Confusion matrix for KNN

cm\_knn = confusion\_matrix(y\_test, y\_pred\_knn)

# KNN decision boundary plot

def label\_name(val):
return "No" if val == 0 else "Yes"

X\_set, y\_set = X\_test\_pca, y\_pred\_knn
X1, X2 = np.meshgrid(
np.arange(X\_set\[:, 0].min() - 1, X\_set\[:, 0].max() + 1, 0.01),
np.arange(X\_set\[:, 1].min() - 1, X\_set\[:, 1].max() + 1, 0.01)
)
plt.contourf(X1, X2, knn\_final.predict(np.array(\[X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha=0.25, cmap=ListedColormap(('cyan', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, val in enumerate(np.unique(y\_test)):
plt.scatter(X\_set\[y\_test == val, 0], X\_set\[y\_test == val, 1],
c=ListedColormap(('blue', 'crimson'))(i), label=label\_name(val), s=10)

plt.title('KNN Decision Boundary')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Confusion matrix plot

plt.figure()
plt.imshow(cm\_knn, interpolation='nearest', cmap=plt.cm.Accent)
plt.title('Confusion Matrix - KNN')
plt.ylabel('True label')
plt.xlabel('Predicted label')
class\_names = \['No', 'Yes']
plt.xticks(np.arange(2), class\_names)
plt.yticks(np.arange(2), class\_names, rotation=45)

labels = \[\['TN', 'FP'], \['FN', 'TP']]
for i in range(2):
for j in range(2):
plt.text(j, i, f"{labels\[i]\[j]} = {cm\_knn\[i]\[j]}", ha='center', va='center')
plt.tight\_layout()
plt.show()

# Decision Tree classification and evaluation

d\_range = range(2, 10)
dtree\_f1, dtree\_js, dtree\_log\_loss = \[], \[], \[]
for d in d\_range:
dtree = DecisionTreeClassifier(criterion='entropy', max\_depth=d)
dtree.fit(X\_train\_pca, y\_train)
y\_pred = dtree.predict(X\_test\_pca)
dtree\_f1.append(f1\_score(y\_test, y\_pred))
dtree\_js.append(jaccard\_score(y\_test, y\_pred))
dtree\_log\_loss.append(log\_loss(y\_test, y\_pred))

# Plot Decision Tree metrics

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = \[(dtree\_f1, 'F1 Score'), (dtree\_js, 'Jaccard'), (dtree\_log\_loss, 'Log Loss')]
colors = \['blue', 'red', 'orange']

for i, (metric, title) in enumerate(metrics):
axes\[i].plot(d\_range, metric, color=colors\[i])
axes\[i].set\_title(f'Decision Tree {title}')
axes\[i].set\_xlabel('Max Depth')
axes\[i].set\_ylabel('Score')
axes\[i].set\_xticks(list(d\_range))

plt.tight\_layout()
plt.show()
