# Face Mask Detection (MobileNetV2 + OpenCV)

Real-time system that detects faces with an OpenCV Haar cascade and classifies each as **Mask / No Mask** using a fine-tuned MobileNetV2 model.

## Results
- Dataset: [dataset name and Kaggle link], [X] face crops
- Validation accuracy: [X]%  |  Precision: [X]  |  Recall: [X]

![Training curves](results/training_curves.png)
![Confusion matrix](results/confusion_matrix.png)
![Demo](results/result1.jpg)

## How it works
1. Haar cascade finds faces in the frame
2. Each face is cropped (20% padding) and resized to 160x160
3. MobileNetV2 classifies it as Mask (green) or No Mask (red)

## Run it
    pip install -r requirements.txt
    python run_mask_detection.py
Press `q` or ESC to exit. Requires a webcam.

## Colab
Open `Face_Mask_Detection.ipynb` in Google Colab to train the model or test it on photos.

## Tech
Python, TensorFlow/Keras, OpenCV, MobileNetV2 (transfer learning)
