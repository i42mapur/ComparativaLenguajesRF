import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Leemos el fichero CSV con los datos y lo almacenamos en la variable "adult"
adult = pd.read_csv("/home/i42mapur/TFM/datos/datos_10000.csv");

# Extraemos las primeras 14 columnas (la clase está en la columna 15)
# Extraemos la columna de clase
clases = np.array(adult.pop('over50K'))

# Transformamos los valores de texto en numéricos
le = preprocessing.LabelEncoder()

categorical_feature_mask = adult.dtypes==object
categorical_cols = adult.columns[categorical_feature_mask].tolist()
adult[categorical_cols] = adult[categorical_cols].apply(lambda col: le.fit_transform(col))

# Creamos el objeto randomForest
rf = RandomForestClassifier(n_estimators=100, max_features = 'sqrt', n_jobs=-1,
verbose = 1, oob_score=True)

# Realizamos el entrenamiento
rf.fit(adult, clases)
rf.oob_score_
