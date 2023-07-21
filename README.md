# Traffic Sign Recognition Project Documentation

## Overview

The Traffic Sign Recognition project aims to detect and recognize traffic signs in real-time using a webcam feed. The project uses Canny edge detection for traffic sign segmentation and a pre-trained CNN model for traffic sign classification. When a traffic sign is detected, the program displays the name of the recognized sign, its class label, and the prediction probability on the webcam feed.

## Requirements

The following libraries are required to run the project:
- Python (>= 3.6)
- OpenCV (>= 4.5)
- TensorFlow (>= 2.0)

You can install the required Python libraries using the following command:
```console
pip install opencv-python tensorflow
```

## Dataset

For training the traffic sign recognition model, the project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The dataset contains images of various traffic signs along with their corresponding class labels.

The GTSRB dataset can be downloaded from the following link:
[GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download)

## Usage

- Ensure that the required Python libraries are installed (OpenCV and TensorFlow).
- Connect your webcam to the computer.
- Run the project script

## Referrences

- [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download) - The German Traffic Sign Recognition Benchmark (GTSRB) dataset used for training the traffic sign recognition model.
- [OpenCV Documentation](https://docs.opencv.org/4.x/index.html) - Documentation for OpenCV library.
- [TensorFlow](https://www.tensorflow.org/api_docs) - Documentation for TensorFlow library.
- [Python 3.11.4 documentation](https://docs.python.org/3/) - Official Python documentation.

## Conclusion

The Traffic Sign Recognition project combines Canny edge detection with a pre-trained CNN model to detect and recognize traffic signs in real-time using a webcam. The project demonstrates the use of computer vision techniques and deep learning for traffic sign recognition, which can be extended to real-world applications such as autonomous vehicles, driver assistance systems, and traffic management.
