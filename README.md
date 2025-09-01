# :tshirt: **Virtual Try-On Lite (VITON Lite)**

---

## :dart: **Objective**

---

Given an image of a person and an image of clothes mock-up, how to apply the mock-up onto the person image? Virtual Try-On (VTO) is a technology which does exactly that. This project aims to answer the objective by building a Virtual Try-On Network (VTON) pipeline that generates the Virtual Try-On image.

## :arrow_right: **The Pipeline Process**

---

![Figure 1. The Pipeline Process](https://drive.google.com/uc?export=view&id=1LInCagyZkMfjGIR_5UictP1jNZ1PAgjN)

The figure above shows the process of the developed pipeline. First, the pose skeleton is estimated with MediaPipe Pose Estimation model with the person image as the input. The person image and the pose skeleton become the inputs for the U-Net model to output the agnostic segmentation, where the clothes region and the background region are left off. Then, the person image is masked with the agnostic segmentation, resulting the agnostic representation. After that, the clothes mock-up image is segmented by a U-Net model. The agnostic representation, the pose skeleton, the clothes mock-up image, and the clothes segmentation are fed into the warping model to generate warped clothes. Final step, the agnostic representation, the pose skeleton, and the synthesized warped clothes become the input for the virtual try-on model to generate the virtual try-on image.

## :building_construction: **Models Architecture**

---

- Additive Attention U-Net is used as the backbone of the segmentation models, the warping model, and the generators.
- Convolutional Neural Network (CNN) for the patch discriminators.

## :hammer_and_wrench: **Tools Used**

---

- Tensorflow : For building and training models
- VoxelMorph : For warping clothes mock-up image
- MediaPipe : For predicting pose landmarks
- Numpy : Used in inference for array manipulation
- OpenCV : Used in inference for load and save images
- Matplotlib : Used for optional visualization

# :rocket: **How to Run the Inference**

---

## :package: **Prerequisites**

---

This repository stores codes and models to run the inference. To be able to run this on your machine, you have to get Python and Git installed. This experiment could run the inference in Python 3.11.13.

## :memo: **Steps**

---

To run the inference, clone this repository with this command, then open the repository folder.

```
git clone https://github.com/WilliamDevin23/virtual-tryon-patchgan.git
```

This project requires libraries listed in requirements.txt. Install them with this command.

```
pip install -r requirements.txt
```

Open the inference folder. There is the _`inference.py`_ script, run it with this command.

```
python inference.py
```

This command will do the inference given the _`example1.jpg`_ and _`cloth_example1.jpg`_ as the inputs. You could replace these images as your desire, either with the same name, or change the _`IMAGE_PATH`_ and _`CLOTH_IMAGE_PATH`_ value in the _`inference.py`_ script, precisely in the lines 9^th^ and 10^th^.

```python
...
# Replace the values with your images name
IMAGE_PATH = 'example1.jpg'
CLOTH_IMAGE_PATH = 'cloth_example1.jpg'
...
```

# :books: **References**

---

1. [MediaPipe Pose Estimation with Python](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
2. [MediaPipe Pose Landmarker Notebook](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb)
3. [VoxelMorph GitHub](https://github.com/voxelmorph/voxelmorph)
4. [VITON-HD Dataset](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset/data)

---
