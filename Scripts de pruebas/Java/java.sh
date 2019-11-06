#Se definen las variables necesarias para que se encuentren las librer�as
export WEKAINSTALL=/home/i42mapur/Descargas/weka-3-8-3
export CLASSPATH=$CLASSPATH:$WEKAINSTALL/weka.jar

#Llamada para generar el modelo
java weka.classifiers.trees.RandomForest -t /home/i42mapur/TFM/datos/datos.arff -d models/java_arff.model -v -O -output-out-of-bag-complexity-statistics -attribute-importance -num-slots 4 -no-cv

#Llamada para aplicar el modelo a un conjunto de datos
java weka.classifiers.trees.RandomForest -T /home/i42mapur/TFM/datos/datos_.arff -l models/java_arff.model
