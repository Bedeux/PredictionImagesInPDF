# Importation des librairies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import json

np.set_printoptions(precision=2, threshold=np.inf)

def initialisation(dimensions):

    parametres = {}

    nbcouche = len(dimensions)
    for couche in range(1, nbcouche):
        parametres['W'+str(couche)] = np.random.randn(dimensions[couche],dimensions[couche-1])
        parametres['b' + str(couche)] = np.random.randn(dimensions[couche], 1)
    return parametres

def forward_propagation(X, parametres):

    activations = {'A0' : X}

    nbCouche = len(parametres) // 2

    for couche in range(1, nbCouche+1):
        Z = parametres['W' + str(couche)].dot(activations['A' + str(couche-1)]) + parametres['b' + str(couche)]
        activations ['A' + str(couche)] = 1 / (1+np.exp(-Z))

    return activations

def back_propagation(y, activations, parametres):

    m = y.shape[1]
    nbCouche = len(parametres) // 2

    dZ = activations['A' + str(nbCouche)] - y
    gradients = {}

    for couche in reversed(range(1,nbCouche+1)):
        gradients['dW' + str(couche)] = 1 / m * np.dot(dZ, activations['A'+str(couche - 1)].T)
        gradients['db' + str(couche)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if couche > 1:
            dZ = np.dot(parametres['W' + str(couche)].T, dZ) * activations['A' + str(couche - 1)] * (1 - activations['A' + str(couche-1)])

    return gradients

def update(parametres, gradients, learning_rate):

    nbCouche = len(parametres) // 2

    for couche in range(1,nbCouche + 1):
        parametres['W' + str(couche)] = np.round(parametres['W' + str(couche)] - learning_rate * gradients['dW' + str(couche)],4)
        parametres['b' + str(couche)] = np.round(parametres['b' + str(couche)] - learning_rate * gradients['db' + str(couche)],4)

    return parametres

def predict(X, parametres):
    activations = forward_propagation(X,parametres)
    nbCouche = len(parametres) // 2
    Af = activations['A'+str(nbCouche)]
    return Af >= 0.5

def neural_network(X_train, Y_train, hidden_layers, X_test, Y_test, learning_rate=0.1, n_iter=1000):

    np.random.seed(0)

    # initialisation
    neural_network_dimension = list(hidden_layers)
    neural_network_dimension.insert(0,X_train.shape[0])
    neural_network_dimension.append(Y_train.shape[0])

    neural_network_parametres = initialisation(neural_network_dimension)

    colonne_param = []
    colonne_accuracy = ['train Loss','test Loss','train accuracy','test accuracy']

    for col in range(1,(len(neural_network_parametres)//2)+1):
        colonne_param.append('W'+str(col))
        colonne_param.append('b' + str(col))

    db_param = pd.DataFrame(columns=colonne_param)
    db_accuracy = pd.DataFrame(columns=colonne_accuracy)

    for i in tqdm(range(n_iter)):
        # Actication
        neural_network_activations = forward_propagation(X_train, neural_network_parametres)
        # Gradients
        gradients = back_propagation(Y_train, neural_network_activations, neural_network_parametres)

        nbCouche = len(neural_network_parametres) // 2
        # Entrainement
        # Calcul du cout
        train_loss = log_loss(Y_train, neural_network_activations['A' + str(nbCouche)])
        # Calcul de l'accuracy
        y_pred_train = predict(X_train, neural_network_parametres)
        train_accuracy = accuracy_score(Y_train.flatten(), y_pred_train.flatten())

        # Test
        # Calcul du cout
        activations_test = forward_propagation(X_test, neural_network_parametres)
        test_loss = log_loss(Y_test, activations_test['A' + str(nbCouche)])
        # Calcul de l'accuracy
        y_pred_test = predict(X_test, neural_network_parametres)
        test_accuracy = accuracy_score(Y_test.flatten(), y_pred_test.flatten())

        # Intégration de toutes les valeurs que l'on souhaite dans une liste
        liste_param = []
        for j in range(1,nbCouche+1):
            liste_param.append(neural_network_parametres["W" + str(j)])
            liste_param.append(neural_network_parametres["b" + str(j)])

        liste_accuracy = []
        liste_accuracy.append(train_loss)
        liste_accuracy.append(test_loss)
        liste_accuracy.append(train_accuracy)
        liste_accuracy.append(test_accuracy)

        db_param.loc[len(db_param)] = liste_param
        db_accuracy.loc[len(db_param)] = liste_accuracy

        # Maj des paramètres
        neural_network_parametres = update(neural_network_parametres, gradients, learning_rate)

    return db_param, db_accuracy

# Lecture du jeu de donnée
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

# Partage du jeu de données en 75,25
X_train, X_test, Y_train, Y_test = train_test_split(bdd2,Target, test_size = 0.25)
Y_train = Y_train.to_numpy().reshape((1,Y_train.shape[0]))
Y_test = Y_test.to_numpy().reshape((1,Y_test.shape[0]))

X_train_Normal = X_train.copy()
X_test_Normal = X_test.copy()

# Normaliser les données
for colonne in X_train.columns[0:5]:
    X_train_Normal[colonne] = (X_train[colonne] - X_train[colonne].min()) / (X_train[colonne].min() + X_train[colonne].max())
    X_test_Normal[colonne] = (X_test[colonne] - X_test[colonne].min()) / (X_test[colonne].min() + X_test[colonne].max())

bdd_parameters,bdd_accuracy = neural_network(X_train_Normal.T,Y_train,(1,20,20,20),X_test_Normal.T,Y_test,learning_rate=0.3, n_iter=10000)

best_pred_index = bdd_accuracy['test accuracy'].idxmax()
print(best_pred_index)
print(bdd_accuracy.loc[best_pred_index])

# # Visualisation des résultats
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(bdd_accuracy['train Loss'])
# plt.plot(bdd_accuracy['test Loss'])
# plt.subplot(1, 2, 2)
# plt.plot(bdd_accuracy['train accuracy'])
# plt.plot(bdd_accuracy['test accuracy'])
# plt.show()

print(bdd_parameters.loc[best_pred_index].to_dict())

best_pred = predict(X_test_Normal.T,parametres=bdd_parameters.loc[best_pred_index])
print(confusion_matrix(Y_test.flatten(), best_pred.flatten()))
dico = {}

g = [0.26582278481012656, 0.040282685512367494, 0.34824756666312096, 0.6803334039461483, 1.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p = [0.10126582278481013, 0.2750294464075383, 0.34824756666312096, 0.36247261051566354, 1.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q = [0.4177215189873418, 0.060306242638398115, 0.34824756666312096, 0.33742457967373596, 0.97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
r = [0.31645569620253167, 0.1988221436984688, 0.34824756666312096, 0.15029759922549082, 0.43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s= [0.02531645569620253, 0.6348645465253239, 0.34824756666312096, 0.23088503742837269, 0.66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t= [0.189873417721519, 0.21319199057714958, 0.34824756666312096, 0.41950644962711864, 1.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(bdd_parameters.loc[best_pred_index].to_dict())
g1 = predict(g,bdd_parameters.loc[best_pred_index].to_dict())
p1 = predict(p,bdd_parameters.loc[best_pred_index].to_dict())
q1 = predict(q,bdd_parameters.loc[best_pred_index].to_dict())
r1 = predict(r,bdd_parameters.loc[best_pred_index].to_dict())
s1 = predict(s,bdd_parameters.loc[best_pred_index].to_dict())
t1 = predict(t,bdd_parameters.loc[best_pred_index].to_dict())
print(g1)
print(p1)
print(q1)
print(r1)
print(s1)
print(t1)