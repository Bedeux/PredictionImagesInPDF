import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('final_data.csv',sep = ',')

data = df.drop(['DocumentName','Target'],axis=1).values
target = df['Target'].values

def KNN(data,target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=0, train_size=0.7)

    scores = []
    values = []

    for i in range(1,10):
      clf = KNeighborsClassifier(n_neighbors=i)
      clf.fit(data_train, target_train)
      score = clf.score(data_test, target_test)
      scores.append(round(score,3))
      values.append(i)
      print("Score (KNN): ",round(score,3), " avec ", i ," neighbours")
    print(values,"   ",scores)

def RandomForest(data,target):
    X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=0)
    scores = []
    values = []
    for i in range(100,500,25):
      clf = RandomForestClassifier(n_estimators=i)
      clf.fit(X_train,y_train)
      y_pred=clf.predict(X_test)
      scores.append(metrics.accuracy_score(y_test, y_pred))
      values.append(i)
      print('Accuracy (Random Forest): ',metrics.accuracy_score(y_test, y_pred))
    print(scores, " ", values)
    print("max score : ",max(scores))

def SVM(data,target):
    # kernels = ["linear", "poly", "rbf", "sigmoid"]
    kernels = ["rbf"]

    X_tr, X_tst, y_tr, y_tst = train_test_split(data, target, test_size=0.3, random_state=0)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_tst)
    print("Accuracy (with rbf): ", round(metrics.accuracy_score(y_tst, y_pr), 3))

def DecisionTree(data,target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 42)

    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    clf_en.fit(X_train, y_train)
    y_pred_en = clf_en.predict(X_test)
    scores = []
    values = []

    for i in range(1,25):
        clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
        clf_en.fit(X_train, y_train)
        y_pred_en = clf_en.predict(X_test)
        scores.append(round(accuracy_score(y_test, y_pred_en),3))
        values.append(i)

    print(scores)
    print(values)

# KNN(data, target)
# SVM(data, target)
# DecisionTree(data,target)
RandomForest(data,target)

