import numpy as np
import pandas as pd
import xgboost as xgb
import time

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# wczytanie i podział danych
catdog_df = pd.read_csv('cechy.csv', sep=',')
X = catdog_df.iloc[:, 1:-1]
Y = catdog_df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50)

print(catdog_df)
print(X_train)
print(X_test)

# wykres cech
pies=catdog_df.loc[catdog_df['Rodzaj']==0]
kot=catdog_df.loc[catdog_df['Rodzaj']==1]
pies=pies.iloc[:,1:-1]
kot=kot.iloc[:,1:-1]
catdog=[pies,kot]
nazwy=["Pies", 'Kot']
width=0.3
X=pies.columns
ind=np.arange(len(X))
for nr,dane in enumerate(catdog):
    Y=[]
    for i,a in enumerate(X):
        Y.append(0)
    for x,row in dane.iterrows():
        for i,kolumna in enumerate(X):
            Y[i]+=row[kolumna]

    plt.bar(ind+nr*width,Y,width, label=f'Zwierzę: {nazwy[nr]}')
    plt.xlabel('cecha')
    plt.ylabel('liczba wystąpień')
    plt.xticks(ind+width/2, X, rotation=45, ha="right")
plt.legend()
plt.title('Liczba wystąpień cech dla psa i kota')
plt.tight_layout()
plt.show()
plt.clf()

# metoda xgboost (drzewo decyzyjne)
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)
param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic'}
num_round = 20
bst = xgb.train(param, dtrain, num_round)
t1 = time.time()
preds1 = bst.predict(dtest)
t2 = time.time()
czas1 = t2 - t1
print('Czas działania predykcji - ' + str(czas1))
a = np.rint(preds1)
accuracy = accuracy_score(Y_test, a) * 100
print('Dopasowanie modelu xgboost jest równe - ' + str(round(accuracy, 2)) + ' %.')
print('Precyzja - ' + str(precision_score(Y_test, a, average=None)))
print('F1 - ' + str(f1_score(Y_test, a, average=None)))
print('Macierz pomyłek - ' + str(confusion_matrix(Y_test, a)))

# wyświetlenie confusion matrix
cm1 = confusion_matrix(Y_test, a)
ax1 = sns.heatmap(cm1, annot=True, cmap='Blues')
ax1.set_xlabel('Przewidziane wartości')
ax1.set_ylabel('Prawdziwe wartości')
ax1.set_title('Macierz pomyłek dla modelu xgboost')
ax1.xaxis.set_ticklabels(['Pies', 'Kot'])
ax1.yaxis.set_ticklabels(['Pies', 'Kot'])
plt.show()

# metoda KNN (najbliższych sąsiadów)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
t3 = time.time()
preds2 = classifier.predict(X_test)
t4 = time.time()
czas2 = t4 - t3
print('Czas działania predykcji - ' + str(czas2))
accuracy = accuracy_score(Y_test, preds2) * 100
print('Dopasowanie modelu KNN jest równe - ' + str(round(accuracy, 2)) + ' %.')
print('Precyzja - ' + str(precision_score(Y_test, preds2, average=None)))
print('F1 - ' + str(f1_score(Y_test, preds2, average=None)))
print('Macierz pomyłek - ' + str(confusion_matrix(Y_test, preds2)))

# confusion matrix KNN
cm2 = confusion_matrix(Y_test, preds2)
ax2 = sns.heatmap(cm2, annot=True, cmap='Blues')
ax2.set_xlabel('Przewidziane wartości')
ax2.set_ylabel('Prawdziwe wartości')
ax2.set_title('Macierz pomyłek dla modelu KNN')
ax2.xaxis.set_ticklabels(['Pies', 'Kot'])
ax2.yaxis.set_ticklabels(['Pies', 'Kot'])
plt.show()

# krzywa uczenia KNN
train_sizes, train_scores, test_scores = learning_curve(estimator=classifier, X=X_train, y=Y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Krzywa uczenia dla KNN')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# sprawdzenie optymalnej liczby sąsiadów
k_list = list(range(1, 20, 1))
cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

pom = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15, 10))
plt.title('Optymalna liczba sąsiadów', fontsize=20, fontweight='bold')
plt.xlabel('Liczba sąsiadów (K)', fontsize=15)
plt.ylabel('Błąd błędnej klasyfikacji', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, pom)
plt.show()

# metoda svc
svm = SVC()
svm.fit(X_train, Y_train)
t5 = time.time()
preds3 = svm.predict(X_test)
t6 = time.time()
czas3 = t6 - t5
print('Czas działania predykcji - ' + str(czas3))
accuracy = accuracy_score(Y_test, preds3) * 100
print('Dopasowanie modelu svc jest równe - ' + str(round(accuracy, 2)) + ' %.')
print('Precyzja - ' + str(precision_score(Y_test, preds3, average=None)))
print('F1 - ' + str(f1_score(Y_test, preds3, average=None)))
print('Macierz pomyłek - ' + str(confusion_matrix(Y_test, preds3)))

# confusion matrix SVC
cm3 = confusion_matrix(Y_test, preds3)
ax3 = sns.heatmap(cm3, annot=True, cmap='Blues')
ax3.set_xlabel('Przewidziane wartości')
ax3.set_ylabel('Prawdziwe wartości')
ax3.set_title('Macierz pomyłek dla modelu SVC')
ax3.xaxis.set_ticklabels(['Pies', 'Kot'])
ax3.yaxis.set_ticklabels(['Pies', 'Kot'])
plt.show()

# krzywa uczenia SVC
train_sizes, train_scores, test_scores = learning_curve(estimator=svm, X=X_train, y=Y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Krzywa uczenia dla SVC')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# połączenie wszystkich trzech metod
combine = preds1 * 0.4 + preds2 * 0.3 + preds3 * 0.3
c = np.rint(combine)
accuracy = accuracy_score(Y_test, c) * 100
print('Precyzja połączonych modeli jest równa - ' + str(round(accuracy, 2)) + ' %.')
print('Precyzja - ' + str(precision_score(Y_test, c, average=None)))
print('F1 - ' + str(f1_score(Y_test, c, average=None)))
print('Macierz pomyłek - ' + str(confusion_matrix(Y_test, c)))

# confusion matrix dla połączonych
cm4 = confusion_matrix(Y_test, c)
ax4 = sns.heatmap(cm4, annot=True, cmap='Blues')
ax4.set_xlabel('Przewidziane wartości')
ax4.set_ylabel('Prawdziwe wartości')
ax4.set_title('Macierz pomyłek dla połączonych modeli')
ax4.xaxis.set_ticklabels(['Pies', 'Kot'])
ax4.yaxis.set_ticklabels(['Pies', 'Kot'])
plt.show()

# porównanie czasów predykcji
t = [czas1, czas2, czas3]
name = ['xgboost','KNN','SVC']
plt.xlabel('Algorytm')
plt.ylabel('Czas (ms)')
plt.title('Czas predykcji algorytmów')
plt.bar(name,t)
plt.show()