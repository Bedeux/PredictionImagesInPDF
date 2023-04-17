# Importation des librairies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Lecture du jeu de donnée
bdd = pd.read_csv("data_AI.csv" , sep = ',', header = 0, index_col = 0)
Target = bdd.pop('Target')
bdd2 = pd.get_dummies(data=bdd,columns=["RGB1","RGB2"])

# Partage du jeu de données en 75,25
Xtrain, Xtest, Ytrain, Ytest = train_test_split(bdd2,Target, test_size = 0.25)

Ytrain = Ytrain.to_numpy().reshape((Ytrain.shape[0],1))
Ytest = Ytest.to_numpy().reshape((Ytest.shape[0],1))

Xtrain2 = Xtrain.copy()
Xtest2 = Xtest.copy()

# Normaliser les données
for colonne in Xtrain2.columns:
    Xtrain2[colonne] = (Xtrain2[colonne] - Xtrain2[colonne].min()) / (Xtrain2[colonne].min() + Xtrain2[colonne].max())
    Xtest2[colonne] = (Xtest2[colonne] - Xtest2[colonne].min()) / (Xtest2[colonne].min() + Xtest2[colonne].max())

# Initialisation des valeurs
def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)

    parametres = {
        'W': W,
        'b': b,
    }
    return parametres

def model(X,parametres):
    Z = X.dot(parametres['W']) + parametres['b']
    A = 1/(1+np.exp(-Z)) #Fonction sigmoid
    # A = (np.exp(2*Z)-1)/(np.exp(2*Z)+1) fonction tangeante hyperbolique
    return A

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A+epsilon) - (1 - y) * np.log(1 - A+epsilon)) # Fonction de cout de la sigmoide
    # return 1/ len(y) * np.sum(np.sum(-y*np.log(A+epsilon))) fonction tangeante hyperbolique

def gradients(A,X,y):
    dW = 1/len(y)*np.dot(X.T, A - y)
    db = 1/len(y)*np.sum(A - y)
    return (dW, db)

def update(dW, db, parametres, learning_rate):
    W = parametres['W'] - learning_rate*dW
    b = parametres['b'] - learning_rate*db

    parametres = {
        'W': W,
        'b': b,
    }

    return parametres

def predict(X,parametres):
    A = model(X,parametres)
    return A >=0.5

def artificial_neuron(X, Y,Xt,Yt, learning_rate = 0.1,n_iter=1000):
    #Initialisation W,b
    parametres = initialisation(Xtrain)

    colonne_param = ['W', 'b']
    colonne_accuracy = ['train Loss', 'test Loss', 'train accuracy', 'test accuracy']

    db_param = pd.DataFrame(columns=colonne_param)
    db_accuracy = pd.DataFrame(columns=colonne_accuracy)

    for i in tqdm(range(n_iter)):
        #Actication
        A = model(X,parametres)
        #Entrainement
        #Calcul du cout
        train_loss = log_loss(A,Y)
        #Calcul de l'accuracy
        y_pred = predict(X, parametres)
        train_accuracy = accuracy_score(Y, y_pred)

        #Test
        # Calcul du cout
        A_test = model(Xt,parametres)
        test_loss = log_loss(A_test, Yt)

        # Calcul de l'accuracy
        y_pred = predict(Xt, parametres)
        test_accuracy = accuracy_score(Yt, y_pred)

        liste_param = []
        liste_param.append(parametres['W'])
        liste_param.append(parametres['b'])

        liste_accuracy = []
        liste_accuracy.append(train_loss)
        liste_accuracy.append(test_loss)
        liste_accuracy.append(train_accuracy)
        liste_accuracy.append(test_accuracy)

        db_param.loc[len(db_param)] = liste_param
        db_accuracy.loc[len(db_param)] = liste_accuracy

        #Maj
        dw, db = gradients(A,X,Y)
        parametres = update(dw,db,parametres, learning_rate)

    return db_param,db_accuracy

bdd_param, bdd_accuracy = artificial_neuron(Xtrain2,Ytrain,Xtest2,Ytest, learning_rate=0.005,n_iter=10000)

best_pred_index = bdd_accuracy['test accuracy'].idxmax()
print(best_pred_index)
print(bdd_accuracy.loc[best_pred_index])

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(bdd_accuracy['train Loss'])
plt.plot(bdd_accuracy['test Loss'])
plt.subplot(1, 2, 2)
plt.plot(bdd_accuracy['train accuracy'])
plt.plot(bdd_accuracy['test accuracy'])
plt.show()

best_pred = predict(Xtest2,parametres=bdd_param.loc[best_pred_index])
print(confusion_matrix(Ytest, best_pred))

