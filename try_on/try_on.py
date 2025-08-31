import tensorflow as tf
import os
import cv2
import numpy as np
import voxelmorph as vxm

from agnostic_segmentation.segmentation import get_person_representations
from clothes_segmentation.segmentation import get_cloth_segmentation

def generate_tryon(image, cloth_image, size='small') :
  person_agnostic, pose_skeleton = get_person_representations(image, size=size)
  cloth_segmentation = get_cloth_segmentation(cloth_image, size=size)

  if size == 'small' :
    cloth_image = cv2.resize(cloth_image, (192, 256))
  else :
    cloth_image = cv2.resize(cloth_image, (384, 512), interpolation=cv2.INTER_LINEAR)
  cloth_image = np.expand_dims(cloth_image, axis=0) / 255

  if size == 'small' :
    small_model_path = os.path.join(os.getcwd(), '..', 'models', 'small')
    warp_unet_small = tf.keras.models.load_model(os.path.join(small_model_path, 'warp_unet.keras'))
    tryon_generator_small = tf.keras.models.load_model(os.path.join(small_model_path, 'tryon_generator.keras'))
    
    deformation_fields = warp_unet_small((person_agnostic, pose_skeleton, cloth_image, cloth_segmentation))
    warped_cloth = vxm.layers.SpatialTransformer()([cloth_image, deformation_fields]).numpy()
    tryon = tryon_generator_small((person_agnostic, pose_skeleton, warped_cloth))
  else :
    large_model_path = os.path.join(os.getcwd(), '..', 'models', 'large')
    warp_unet_large = tf.keras.models.load_model(os.path.join(large_model_path, 'warp_unet_large.keras'))
    tryon_generator_large = tf.keras.models.load_model(os.path.join(large_model_path, 'tryon_generator_large.keras'))
    
    deformation_fields = warp_unet_large((person_agnostic, pose_skeleton, cloth_image, cloth_segmentation))
    warped_cloth = vxm.layers.SpatialTransformer()([cloth_image, deformation_fields]).numpy()
    tryon = tryon_generator_large((person_agnostic, pose_skeleton, warped_cloth))
  return tryon[0]