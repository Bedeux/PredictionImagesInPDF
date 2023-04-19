# Importation des librairies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def initialisation(n0,n1,n2):
    W1 = np.random.randn(n1,n0)
    b1 = np.random.randn(n1,1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parametres = {
        'W1' : W1,
        'b1' : b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def forward_propagation(X,parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1' : A1,
        'A2' : A2
    }

    return activations

def back_propagation(X, y, activations, parametres):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]
    # Calcul des Gradients
    # Première couche
    dZ2 = A2 - y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    #Deuxième couche
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

def update(parametres,gradients, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1' : W1,
        'b1' : b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def predict(X, parametres):
    activations = forward_propagation(X,parametres)
    A2 = activations["A2"]
    return A2 >= 0.5

def neural_network(X_train,Y_train,nb_neuronne_couche1,X_test,Y_test,learning_rate = 0.1,n_iter = 1000):

    n0 = X_train.shape[0]
    n2 = Y_train.shape[0]
    n1 = nb_neuronne_couche1
    # Initialisation W1, b1, W2, b2
    parametres = initialisation(n0,n1,n2)

    colonne_param = ['W1','b1','W2','b2']
    colonne_accuracy = ['train Loss', 'test Loss', 'train accuracy', 'test accuracy']

    db_param = pd.DataFrame(columns=colonne_param)
    db_accuracy = pd.DataFrame(columns=colonne_accuracy)

    for i in tqdm(range(n_iter)):
        # Actication
        activations = forward_propagation(X_train, parametres)

        # Gradients
        gradients = back_propagation(X_train,Y_train,activations,parametres)

        # Entrainement
        # Calcul du cout
        train_loss_ = log_loss(Y_train, activations['A2'])
        # Calcul de l'accuracy
        y_pred = predict(X_train, parametres)
        accuracy_train = accuracy_score(Y_train.flatten(), y_pred.flatten())

        # Test
        # Calcul du cout
        activations_test = forward_propagation(X_test, parametres)
        test_loss_ = log_loss(Y_test, activations_test['A2'])
        # Calcul de l'accuracy
        y_pred = predict(X_test, parametres)
        accuracy_test = accuracy_score(Y_test.flatten(), y_pred.flatten())

        # Intégration de toutes les valeurs que l'on souhaite dans une liste
        liste_param = []
        liste_param.append(parametres['W1'])
        liste_param.append(parametres['b1'])
        liste_param.append(parametres['W2'])
        liste_param.append(parametres['b2'])

        liste_accuracy = []
        liste_accuracy.append(train_loss_)
        liste_accuracy.append(test_loss_)
        liste_accuracy.append(accuracy_train)
        liste_accuracy.append(accuracy_test)

        db_param.loc[len(db_param)] = liste_param
        db_accuracy.loc[len(db_param)] = liste_accuracy

        # Maj des paramètres
        parametres = update(parametres, gradients, learning_rate)

    return db_param, db_accuracy

# Lecture du jeu de donnée1
#bdd = pd.read_csv("final_data.csv" , sep = ',', header = 0, index_col = 0)
bdd = pd.read_csv("data_AI.csv" , sep = ',', header = 0, index_col = 0)
Target = bdd.pop('Target')
colonne_couleur = ["(128, 0, 0)","(139, 0, 0)","(165, 42, 42)","(178, 34, 34)","(220, 20, 60)","(255, 0, 0)","(255, 99, 71)",
                   "(255, 127, 80)","(205, 92, 92)","(240, 128, 128)","(233, 150, 122)","(250, 128, 114)","(255, 160, 122)",
                   "(255, 69, 0)","(255, 140, 0)","(255, 165, 0)","(255, 215, 0)","(184, 134, 11)","(218, 165, 32)","(238, 232, 170)",
                   "(189, 183, 107)","(240, 230, 140)","(128, 128, 0)","(255, 255, 0)","(154, 205, 50)","(85, 107, 47)","(107, 142, 35)",
                   "(124, 252, 0)","(127, 255, 0)","(173, 255, 47)","(0, 100, 0)","(0, 128, 0)","(34, 139, 34)","(0, 255, 0)",
                   "(50, 205, 50)","(144, 238, 144)","(152, 251, 152)","(143, 188, 143)","(0, 250, 154)","(0, 255, 127)",
                   "(46, 139, 87)","(102, 205, 170)","(60, 179, 113)","(32, 178, 170)","(47, 79, 79)","(0, 128, 128)","(0, 139, 139)",
                   "(0, 255, 255)","(0, 255, 255)","(224, 255, 255)","(0, 206, 209)","(64, 224, 208)","(72, 209, 204)","(175, 238, 238)",
                   "(127, 255, 212)","(176, 224, 230)","(95, 158, 160)","(70, 130, 180)","(100, 149, 237)","(0, 191, 255)",
                   "(30, 144, 255)","(173, 216, 230)","(135, 206, 235)","(135, 206, 250)","(25, 25, 112)","(0, 0, 128)","(0, 0, 139)",
                   "(0, 0, 205)","(0, 0, 255)","(65, 105, 225)","(138, 43, 226)","(75, 0, 130)","(72, 61, 139)","(106, 90, 205)",
                   "(123, 104, 238)","(147, 112, 219)","(139, 0, 139)","(148, 0, 211)","(153, 50, 204)","(186, 85, 211)",
                   "(128, 0, 128)","(216, 191, 216)","(221, 160, 221)","(238, 130, 238)","(255, 0, 255)","(218, 112, 214)","(199, 21, 133)",
                   "(219, 112, 147)","(255, 20, 147)","(255, 105, 180)","(255, 182, 193)","(255, 192, 203)","(250, 235, 215)",
                   "(245, 245, 220)","(255, 228, 196)","(255, 235, 205)","(245, 222, 179)","(255, 248, 220)","(255, 250, 205)",
                   "(250, 250, 210)","(255, 255, 224)","(139, 69, 19)","(160, 82, 45)","(210, 105, 30)","(205, 133, 63)","(244, 164, 96)",
                   "(222, 184, 135)","(210, 180, 140)","(188, 143, 143)","(255, 228, 181)","(255, 222, 173)","(255, 218, 185)",
                   "(255, 228, 225)","(255, 240, 245)","(250, 240, 230)","(253, 245, 230)","(255, 239, 213)","(255, 245, 238)",
                   "(245, 255, 250)","(112, 128, 144)","(119, 136, 153)","(176, 196, 222)","(230, 230, 250)","(255, 250, 240)",
                   "(240, 248, 255)","(248, 248, 255)","(240, 255, 240)","(255, 255, 240)","(240, 255, 255)","(255, 250, 250)",
                   "(0, 0, 0)","(105, 105, 105)","(128, 128, 128)","(169, 169, 169)","(192, 192, 192)","(211, 211, 211)",
                   "(220, 220, 220)","(245, 245, 245)","(255, 255, 255)"]

dico_couleur_1 = {}
dico_couleur_2 = {}
dico_couleur = {}

for couleur in colonne_couleur:
    dico_couleur_1[couleur] = [1 if couleur == row else 0 for row in bdd['RGB1']]
    dico_couleur_2[couleur] = [1 if couleur == row else 0 for row in bdd['RGB2']]

db_couleur_1 = pd.DataFrame(dico_couleur_1)
db_couleur_2 = pd.DataFrame(dico_couleur_2)

for colonne in db_couleur_1.columns:
    db_couleur_1 = db_couleur_1.rename(columns={colonne:"RGB1" + str(colonne)})
    db_couleur_2 = db_couleur_2.rename(columns={colonne:"RGB2" + str(colonne)})

db_couleur = db_couleur_1.join(db_couleur_2)
db_couleur = db_couleur.set_index(bdd.index)
bdd2 = pd.concat([bdd, db_couleur], axis=1)

bdd2 = bdd2.drop(columns=["RGB1","RGB2"])
#bdd2 = pd.get_dummies(data=bdd,columns=["RGB1","RGB2"])

# Partage du jeu de données en 75,25
#X_train, X_test, Y_train, Y_test = train_test_split(bdd.iloc[:,0:12],bdd['Target'], test_size = 0.25)
X_train, X_test, Y_train, Y_test = train_test_split(bdd2,Target, test_size = 0.25)
Y_train = Y_train.to_numpy().reshape((1,Y_train.shape[0]))
Y_test = Y_test.to_numpy().reshape((1,Y_test.shape[0]))

X_train_Normal = X_train.copy()
X_test_Normal = X_test.copy()

# Normaliser les données
for colonne in X_train.columns[0:5]:
    X_train_Normal[colonne] = (X_train[colonne] - X_train[colonne].min()) / (X_train[colonne].min() + X_train[colonne].max())
    X_test_Normal[colonne] = (X_test[colonne] - X_test[colonne].min()) / (X_test[colonne].min() + X_test[colonne].max())

X_train_Normal = X_train_Normal.T
X_test_Normal = X_test_Normal.T

bdd_parameters,bdd_accuracy = neural_network(X_train_Normal,Y_train,5,X_test_Normal,Y_test,learning_rate=0.2, n_iter=10000)

best_pred_index = bdd_accuracy['test accuracy'].idxmax()
print(best_pred_index)
print(bdd_accuracy.loc[best_pred_index])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(bdd_accuracy['train Loss'])
plt.plot(bdd_accuracy['test Loss'])
plt.subplot(1, 2, 2)
plt.plot(bdd_accuracy['train accuracy'])
plt.plot(bdd_accuracy['test accuracy'])
plt.show()

best_pred = predict(X_test_Normal,parametres=bdd_parameters.loc[best_pred_index])
print(confusion_matrix(Y_test.flatten(), best_pred.flatten()))