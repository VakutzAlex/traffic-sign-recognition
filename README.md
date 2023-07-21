Traffic Sign Recognition Project Documentation
Overview
The Traffic Sign Recognition project aims to detect and recognize traffic signs in real-time using a webcam feed. The project uses Canny edge detection for traffic sign segmentation and a pre-trained CNN model for traffic sign classification. When a traffic sign is detected, the program displays the name of the recognized sign, its class label, and the prediction probability on the webcam feed.

Table of Contents
Overview
Table of Contents
Requirements
Dataset
Installation
Usage
Project Structure
References
Requirements
The following libraries are required to run the project:

Python (>= 3.6)
OpenCV (>= 4.5)
TensorFlow (>= 2.0)
You can install the required Python libraries using the following command:

bash
Copy code
pip install opencv-python tensorflow
Dataset
For training the traffic sign recognition model, the project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The dataset contains images of various traffic signs along with their corresponding class labels.

The GTSRB dataset can be downloaded from the following link:
GTSRB Dataset

Installation
Clone the project repository to your local machine:
bash
Copy code
git clone https://github.com/your_username/traffic-sign-recognition.git
Navigate to the project directory:
bash
Copy code
cd traffic-sign-recognition
Download the GTSRB dataset and organize it in the following structure:
r
Copy code
traffic-sign-recognition
│─── pretrained_model.h5
│─── traffic_sign_recognition.py
└─── test_images
Note: Place the pre-trained model file (pretrained_model.h5) and the traffic_sign_recognition.py script in the project root directory.

Usage
Ensure that the required Python libraries are installed (OpenCV and TensorFlow).

Connect your webcam to the computer.

Run the project script:

bash
Copy code
python traffic_sign_recognition.py
The webcam will start, and the program will begin detecting and recognizing traffic signs in real-time. The name of the recognized sign, its class label, and prediction probability will be displayed on the webcam feed.

Press 'q' to stop the real-time detection and close the webcam window.

Project Structure
The project directory contains the following files:

pretrained_model.h5: The pre-trained model file containing the weights and architecture of the CNN model for traffic sign recognition.

traffic_sign_recognition.py: The Python script for real-time traffic sign recognition using Canny edge detection and the pre-trained CNN model.

test_images: A directory where you can place images for testing the traffic sign recognition on individual images. This is not used during the webcam detection process.

References
GTSRB Dataset - The German Traffic Sign Recognition Benchmark (GTSRB) dataset used for training the traffic sign recognition model.

OpenCV Documentation - Documentation for OpenCV library.

TensorFlow Documentation - Documentation for TensorFlow library.

Python Official Documentation - Official Python documentation.

Conclusion
The Traffic Sign Recognition project combines Canny edge detection with a pre-trained CNN model to detect and recognize traffic signs in real-time using a webcam. The project demonstrates the use of computer vision techniques and deep learning for traffic sign recognition, which can be extended to real-world applications such as autonomous vehicles, driver assistance systems, and traffic management.
