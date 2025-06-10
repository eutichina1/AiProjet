import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Données 
X = np.array([
    [12, 14, 13, 0],
    [15, 16, 14, 1],
    [9, 10, 8, 1],
    [17, 18, 16, 0],
    [10, 11, 9, 0],
    [14, 15, 13, 1],
    [8, 9, 7, 0],
    [13, 14, 12, 1],
    [11, 12, 11, 0],
    [16, 17, 15, 1],
    [13, 13, 12, 0],
    [10, 11, 10, 1],
    [14, 13, 14, 0],
    [15, 14, 13, 1],
    [12, 13, 12, 0],
])
y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0])  # 1 = admis, 0 = refusé

#  Normalisation des données
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#   modèle
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # sortie binaire
])

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

#  Évaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Précision sur les données de test : {acc*100:.2f}%")

#  Courbes d'apprentissage
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title("Courbe d'accuracy")
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#  Fonction de prédiction
def prediction_etudiant(moyenne, bac, test, sexe):
    entree = np.array([[moyenne, bac, test, sexe]])
    entree_scaled = scaler.transform(entree)
    prediction = model.predict(entree_scaled)
    return "Admis " if prediction[0][0] > 0.5 else "Refusé "

# prédictions
print(prediction_etudiant(14, 15, 13, 1))  
