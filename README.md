# Proyecto: Clasificación de plantas

## Intrucciones:

El archivo que deben usar para realizar la predicción es "run_model_to_predict.py".

Recuerden tomar la foto con un fondo similar al de las imágenes del dataset y en horizontal. El código ya tiene la función de resize para que, independiente con que cámara tomen la foto, pueda entrar a la red neuronal.

El código está hecho para que imprima healthy o not-healthy como predicción. Pueden modificarlo para que les de una variable de salida con un 1 o un 0 simplemente cambiando lo que está dentro del if...else.

El archivo "plants_classification.py" es con el que se entreno la red. Sin emabrgo, ese ya no necesitan correrlo, ya que guarde el modelo ya entrenado en el archivo "plants_trainedNN.h5", que es el que se carga en el archivo donde harán la predicción.

El modelo alcanza poco más del **98%** de exactitud (accuracy).

Las imágenes:
* image_h#
* image_n#

son solo muestras que utilice para hacer pruebas. Deben cambiar la variable img_dir con la dirección donde se guardó la imagen que desean introducir.
