import  tensorflow as tf
import numpy as np
import os
import cv2

model = tf.keras.models.load_model(os.path.join(os.getcwd(), "..", "models", "clothes_segmentation.keras"))

def get_cloth_segmentation(cloth_image, size='small') :
  if size not in ['small', 'large'] :
    raise ValueError('Argument size must be either "small" or "large"')
  else :
    cloth_image = cv2.resize(cloth_image, (192, 256))
    cloth_image = np.expand_dims(cloth_image, axis=0) / 255
    cloth_seg = model(cloth_image)
    
    cloth_seg = np.where(cloth_seg > 0.5, 1, 0).astype(np.float32)

    if size != 'small' :
        cloth_seg = cv2.resize(cloth_seg[0], (384, 512), interpolation=cv2.INTER_NEAREST)
        cloth_seg = np.expand_dims(np.expand_dims(cloth_seg, axis=-1), axis=0)
  
    return cloth_seg