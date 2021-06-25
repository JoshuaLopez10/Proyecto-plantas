#%% Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

#%% Caregar el conjunto de datos
x = np.load('images.npy')
y = np.load('labels.npy')

#%% Preprocesar los datos
x = x.astype('float32') / 255
imgs, height, width, bands = x.shape

#%% Separar datos de entrenamiento y  validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 13)

#%% Crear arquitectura de la red
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(height, width, bands)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#%% Compilar la red
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%% Prepare visualizer
callbacks_tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'GraphConv',
                                                       write_images = True,
                                                       write_graph = True,
                                                       histogram_freq = 1)

#%% Entrenar la red
history = model.fit(x_train, y_train, epochs=20, 
                    batch_size=64, 
                    validation_split  = 0.1)

#%% Plot results
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']
plt.plot(accuracy, 'b', label='Training accuracy')
plt.plot(val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% predict with new data
model.predict(x_test)
model.evaluate(x_test, y_test)

#%%
model.save("plants_trainedNN.h5")
