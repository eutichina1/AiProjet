# 🎓 Projet IA – Prédiction d'admission des étudiants

Ce projet est une démonstration d’un modèle d’intelligence artificielle permettant de **prédire si un étudiant est admis ou refusé** sur la base de ses résultats scolaires et de son sexe.  
Le modèle est construit avec **TensorFlow/Keras**, et utilise des techniques de normalisation, d'entraînement supervisé et d'évaluation de performance.

---

## 📌 Objectif du projet

Prédire automatiquement l'admission d'un étudiant à partir de 4 variables :
- Moyenne générale
- Note au bac
- Résultat à un test d’entrée
- Sexe (0 = fille, 1 = garçon)

La prédiction est binaire : `1 = Admis`, `0 = Refusé`.

---

## 🧠 Technologies utilisées

- **Python 3**
- **NumPy**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Matplotlib**

---

## 🔍 Description du modèle

Le réseau neuronal est composé de :
- Une couche d'entrée de 16 neurones avec fonction d'activation ReLU
- Une couche cachée de 8 neurones avec ReLU
- Deux couches `Dropout` pour éviter le surapprentissage
- Une couche de sortie avec une fonction sigmoïde pour une classification binaire

Le modèle est compilé avec :
```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
