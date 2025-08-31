# **Virtual Try-On Lite (VITON Lite)**

---

## **Objective**

---

Given an image of a person and an image of clothes mock-up, how to apply the mock-up onto the person image? Virtual Try-On (VTO) is a technology which does exactly that. This project aims to answer the objective by building a Virtual Try-On Network (VTON) pipeline that generates the Virtual Try-On image.

## **The Pipeline Process**

---

![Figure 1. The Pipeline Process](https://drive.google.com/uc?export=view&id=1LInCagyZkMfjGIR_5UictP1jNZ1PAgjN)

The figure above shows the process of the developed pipeline. First, the pose skeleton is estimated with MediaPipe Pose Estimation model with the person image as the input. The person image and the pose skeleton become the inputs for the U-Net model to output the agnostic segmentation, where the clothes region and the background region are left off. Then, the person image is masked with the agnostic segmentation, resulting the agnostic representation. After that, the clothes mock-up image is segmented by a U-Net model. The agnostic representation, the pose skeleton, the clothes mock-up image, and the clothes segmentation are fed to the warping model to generate warped clothes. Final step, the agnostic representation, the pose skeleton, and the synthesized warped clothes become the input for the virtual try-on model to generate the virtual try-on image.

## **Models Architecture**

---

- Additive Attention U-Net is used as the backbone of the segmentation models, the warping model, and the generators.
- Convolutional Neural Network (CNN) for the patch discriminators.

## **Tools Used**

---

- Tensorflow : For building and training models
- VoxelMorph : For warping clothes mock-up image
- MediaPipe : For predicting pose landmarks
- Numpy : Used in inference for array manipulation
- OpenCV : Used in inference for load and saving images
- Matplotlib : Used for optional visualization
