# Cargamos la librería
library("randomForest");

# Leemos el fichero CSV con los datos y lo almacenamos en la variable "adult"
adult <- read.csv("/home/i42mapur/TFM/datos/datos_7500.csv");

# Extraemos las primeras 14 columnas (la clase está en la columna 15)
atribs <- adult[1:14];

# Extraemos la columna de clase
clase <- adult[15];

# Necesitamos una variable con una lista de valores en vez de una variable con
# muchas listas con un solo valor cada una
cl <- clase[,1];

# Llamamos al método
rf <- randomForest(atribs, cl);

# Vemos el resultado de la clasificación
rf
