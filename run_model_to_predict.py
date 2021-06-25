from keras import models
from PIL import Image
from cv2 import imread
from numpy import array

new_model = models.load_model('plants_trainedNN.h5')

img_dir = 'image_h3.png'
image = array(imread(img_dir))
image.resize(375, 250, 3)
image = image.reshape(1, 375, 250, 3).astype('float32') / 255

y = new_model.predict(image)

if y < 0.5:
    print('healthy')
else:
    print('not_healthy')
