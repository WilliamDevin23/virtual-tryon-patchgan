import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

def get_detector() :
  model_path = os.path.join(os.getcwd(), "..", "models", "pose_landmarker.task")
  base_options = python.BaseOptions(model_asset_path=model_path)
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  
  return detector

def draw_landmarks_on_image(image_size, detection_result, image=None):
  pose_landmarks_list = detection_result.pose_landmarks
  if image is None :
      canvas = np.zeros(image_size)
  else :
      canvas = np.copy(image)

  for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]
      
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        canvas,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
  
  return canvas

def get_mediapipe_skeleton(image) :
  detector = get_detector()
  image = image.astype(np.uint8)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
  detection_result = detector.detect(mp_image)
  landmark = draw_landmarks_on_image(image.shape, detection_result)
  landmark = np.expand_dims(np.mean(landmark, axis=-1), axis=-1)
  
  return tf.cast(landmark / 255, dtype=tf.float32)