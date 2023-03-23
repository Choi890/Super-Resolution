import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image


image_path = 'D:\Code\super_resolution-master\img_align_celeba\img_align_celeba'

def upscale_image(image_path, scale):
    with Image.open(image_path) as im:
        width, height = im.size
        new_width, new_height = width * scale, height * scale
        im = im.resize((new_width, new_height), resample=Image.BILINEAR)
        return im

def upscale_image_tensorflow(image_path, scale):
    model = keras.Sequential([
        keras.Input(shape=(None, None, 3)),
        keras.layers.Lambda(lambda x: tf.image.resize(x, (tf.shape(x)[1] * scale, tf.shape(x)[2] * scale), method='bilinear', antialias=True)),
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    img = model.predict(img)
    img = np.squeeze(img, axis=0)
    img = img * 255.0
    img = img.astype(np.uint8)
    return img

upscaled_image = upscale_image_tensorflow('image.jpg', 2)
plt.imshow(upscaled_image)
plt.show()