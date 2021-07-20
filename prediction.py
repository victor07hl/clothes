#importando librerias

import tensorflow  as tf
import numpy as np
import matplotlib.pyplot as plt
import gzip


#cargando data desde fashion_mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Cargando modelo previamente entrenado
prediction = tf.keras.models.load_model('modelos/predict_clothes.h5')
print('modelo Cargado')

#seleccionado la imagen analizar
#puedes cambiar la  variable num_sample para probar otra imagen
num_sample = 1
new_img = test_images[num_sample]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

true_label = class_names[test_labels[num_sample]]


#construendo grafica con imagen de prueba
plt.figure(1)
plt.imshow(new_img,cmap=plt.cm.binary)

# agregando imagen a lote como lo pide el modelo
# el lote debe ser de 3 dimensiones
new_img = (np.expand_dims(new_img,0))

print('tamaño de la nueva imagen',new_img.shape)

#Realizando la prediccion con el modelo cargado
predictions_single = prediction.predict(new_img/255)
predictions_array  = predictions_single[0]

#obteniendo la etiqueta para la imagen que predice el modelo
predited_label = class_names[np.argmax(predictions_single[0])]

#construyendo grafico final con la etiqueta asignado por el modelo
#cuando la etiqueta predicha es igual a la etiqueta verdadera, 
#el mensaje apparece en verde , de lo contrario en rojo
if predited_label == true_label:
	color = 'green'
else:
	color = 'red'

plt.xlabel("{} ({})".format(predited_label,
							class_names[test_labels[num_sample]]),
							color = color)
plt.title(true_label)

plt.show()
