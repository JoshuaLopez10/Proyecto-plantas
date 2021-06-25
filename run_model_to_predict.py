#%%
from keras import models
from cv2 import imread
from numpy import array

new_model = models.load_model('plants_trainedNN.h5')

img_dir = 'image_h4.png'
image = array(imread(img_dir))
image.resize(375, 250, 3)
image = image.reshape(1, 375, 250, 3)

#%%
y = new_model.predict(image)

if y == 0:
    print('healthy')
else:
    print('not_healthy')
