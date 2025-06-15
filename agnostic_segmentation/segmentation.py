import tensorflow as tf
import numpy as np
import os

AGNOSTIC_CLASS_DICT = {
    (0, 0, 0): 0, # Background
    (0, 0, 254): 1, # Face
    (0, 85, 85): 2, # Pants
    (0, 254, 254): 3, # Right hand
    (51, 169, 220): 4, # Left hand
    (169, 254, 85): 5, # Right thigh
    (85, 254, 169): 6, # Left thigh
    (254, 0, 0): 7 # Hair
}

model = tf.keras.models.load_model(os.path.join(os.getcwd(), '..', 'models', 'agnostic_segmentation.keras'))

def agnostic_segmentation_inference(image) :
  pred = model.predict(image)
  pred = np.argmax(pred, axis=-1)
  class_arr = np.array([pixel for pixel in AGNOSTIC_CLASS_DICT.keys()])
  return class_arr[pred]

def combine(image, segmentation) :
  segmentation = np.where(segmentation == (0, 0, 0), 0, 1)
  return image * segmentation