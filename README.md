# Tennis Ball Detection using Jetson Inference

This project demonstrates how to collect, train, and deploy a custom object detection model for detecting tennis balls using NVIDIA Jetson and the Jetson Inference framework.

---

## Overview

- Collected 2,500 tennis ball images from the Open Images Dataset  
- Trained a custom object detection model using SSD-Mobilenet  
- Converted the trained PyTorch model to ONNX for deployment  
- Tested the model with live video feed and image inputs  
- Retrained the model with additional data and epochs for improved accuracy  

---

## Dataset Collection

The dataset was collected from the Open Images Dataset using the Jetson Inference downloader tool.

### Steps to Collect Tennis Ball Images

1. Search for **"Tennis ball"** on the Open Images Dataset to verify available images.  
2. Connect to the Jetson Orin device using VS Code.  
3. Open terminal and run:

   cd ~/jetson-inference/  
   ./docker/run.sh  

4. Navigate to the detection SSD training folder:

   cd python/training/detection/ssd  

5. Use the downloader script to collect images:

   python3 open_images_downloader.py --max-images=2500 --class-names "Tennis ball" --data=data/tennis_ball

---

## Model Training

### Initial Training

1. Open terminal in VS Code and run the Docker container:

   cd ~/jetson-inference/  
   ./docker/run.sh  

2. Navigate to the SSD training directory:

   cd python/training/detection/ssd  

3. Run the training script:

   python3 train_ssd.py --dataset-type=voc --data=data/tennis_ball --model-dir=models/tennis_ball

---

## Convert PyTorch Model to ONNX

After training is complete, export the model:

   python3 onnx_export.py --model-dir=models/tennis_ball

---

## Model Deployment and Testing

1. Connect to the Jetson device via NoMachine or terminal.  
2. Set the model path:

   NET=~/jetson-inference/python/training/detection/ssd/models/tennis_ball

3. Run the live video detection:

   detectnet --model=$NET/ssd-mobilenet.onnx \  
             --labels=$NET/labels.txt \  
             --input-blob=input_0 \  
             --output-cvg=scores \  
             --output-bbox=boxes \  
             /dev/video0  

---

## Retraining for Improved Performance

The model was retrained using additional images and more training epochs to improve accuracy.  
Follow the same training and deployment steps with the updated dataset.
