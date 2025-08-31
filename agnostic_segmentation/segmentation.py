import tensorflow as tf
import numpy as np
import os
import cv2
from pose_skeleton.extract_pose import get_mediapipe_skeleton

model = tf.keras.models.load_model(os.path.join(os.getcwd(), '..', 'models', 'agnostic_segmentation.keras'))

def get_person_representations(image, size='small') :
  if size not in ['small', 'large'] :
    raise ValueError('Argument size must be either "small" or "large"')
  else :
    # standardize input shape
    image_small = cv2.resize(image, (192, 256))
    image_small = np.expand_dims(image_small, axis=0) / 255

    image = cv2.resize(image, (384, 512))
    image = np.expand_dims(image, axis=0) / 255

    # get pose skeleton
    pose_skeleton = get_mediapipe_skeleton(image_small[0])
    pose_skeleton = np.expand_dims(pose_skeleton, axis=0)

    # predict agnostic segmentation
    agnostic_segmentation = model((image_small, pose_skeleton))
    agnostic_segmentation = np.argmax(agnostic_segmentation, axis=-1)
    agnostic_segmentation = np.expand_dims(np.where(agnostic_segmentation == 0, 0, 1), axis=-1).astype(np.float32)

    if size != 'small' :
      agnostic_segmentation = np.expand_dims(cv2.resize(agnostic_segmentation[0], (384, 512), interpolation=cv2.INTER_NEAREST), axis=-1)
      agnostic_segmentation = np.expand_dims(agnostic_segmentation, axis=0)
      agnostic_representation = image * agnostic_segmentation
      pose_skeleton = np.expand_dims(get_mediapipe_skeleton(image[0]), axis=0)
    else :
      agnostic_representation = image_small * agnostic_segmentation
  return agnostic_representation, pose_skeleton