import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Leemos el fichero CSV con los datos y lo almacenamos en la variable "adult"
adult = pd.read_csv("/home/i42mapur/TFM/datos/datos_40000.csv");

# Extraemos las primeras 14 columnas (la clase está en la columna 15)
# Extraemos la columna de clase
clases = np.array(adult.pop('over50K'))

# Transformamos los valores de texto en numéricos
le = preprocessing.LabelEncoder()

categorical_feature_mask = adult.dtypes==object
categorical_cols = adult.columns[categorical_feature_mask].tolist()
adult[categorical_cols] = adult[categorical_cols].apply(lambda col: le.fit_transform(col))

# Cargamos el modelo
import joblib
rf = joblib.load("/home/i42mapur/TFM/classification/Python/models/python_40000.model")

# Clasificamos los datos en el modelo
rf_predicciones = rf.predict(adult)
from sklearn.metrics import confusion_matrix
confusion_matrix(clases, rf_predicciones)
