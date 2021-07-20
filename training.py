#importando librerias

import tensorflow  as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Revisando el tamaño de la data
print(train_images.shape)
samples = train_images.shape[0]
widht = train_images.shape[1]
height = train_images.shape[2]
print(f"El lote de entrenamiento se compone de {samples} instancias")
print(f"El tamaño de cada imagen es de {widht}X{height} pixeles")
print(f"Los valores de  las etiquetas van desde 0-9 clasificando de esta forma 10 prendas diferentes")

print(f"el tamaño de la data de testeo es de {test_images.shape}")

#Normalizando la data a valores entre 0 y 1

train_images = train_images / 255.0 #manejando imagen de 8 bits
test_images = test_images / 255.0

#.......................................................................................
#Construyendo el modelo con tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#compilando el modelo 
#tomando como optimizador 'adam'
#funcion para medir perdidas 'SparseCategoricalCrossentropy'

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Entrenando modelo

model.fit(train_images, train_labels,epochs=10)
print('pasa')


#Evaluando la  exactitud del modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


#Guardando modelo
model.save('modelos/predict_clothes.h5')

