import cv2
import numpy as np
from keras.models import load_model

# Constants
IMG_SIZE = 30

def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return image

def create_model():
    model = load_model("Traffic.h5")
    return model

def get_traffic_sign_names():
    traffic_sign_names = {
        0: "Speed Limit 20 km/h",
        1: "Speed Limit 30 km/h",
        2: "Speed Limit 50 km/h",
        3: "Speed Limit 60 km/h",
        4: "Speed Limit 70 km/h",
        5: "Speed Limit 80 km/h",
        6: "End of Speed Limit 80 km/h",
        7: "Speed Limit 100 km/h",
        8: "Speed Limit 120 km/h",
        9: "No Passing",
        10: "No Passing for Vehicles over 3.5 metric tons",
        11: "Right-of-Way at the Next Intersection",
        12: "Priority Road",
        13: "Yield",
        14: "Stop",
        15: "No Vehicles",
        16: "Vehicles over 3.5 metric tons prohibited",
        17: "No Entry",
        18: "General Caution",
        19: "Dangerous Curve to the Left",
        20: "Dangerous Curve to the Right",
        21: "Double Curve",
        22: "Bumpy Road",
        23: "Slippery Road",
        24: "Road Narrows on the Right",
        25: "Road Work",
        26: "Traffic Signals",
        27: "Pedestrians",
        28: "Children Crossing",
        29: "Bicycles Crossing",
        30: "Beware of Ice/Snow",
        31: "Wild Animals Crossing",
        32: "End of All Speed and Passing Limits",
        33: "Turn Right Ahead",
        34: "Turn Left Ahead",
        35: "Ahead Only",
        36: "Go Straight or Right",
        37: "Go Straight or Left",
        38: "Keep Right",
        39: "Keep Left",
        40: "Roundabout Mandatory",
        41: "End of No Passing",
        42: "End of No Passing by Vehicles over 3.5 metric tons"
    }
    return traffic_sign_names


def main():
    # Load the pre-trained model and traffic sign names
    model = create_model()
    traffic_sign_names = get_traffic_sign_names()

    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour and make predictions on the detected traffic signs
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust the minimum contour area as needed
                x, y, w, h = cv2.boundingRect(contour)
                sign_image = frame[y:y + h, x:x + w]
                input_image = preprocess_image(sign_image)
                input_image = np.expand_dims(input_image, axis=0)

                # Make a prediction on the sign image using the pre-trained model
                prediction = model.predict(input_image)
                predicted_class = np.argmax(prediction)
                probability = prediction[0][predicted_class]

                # Get the traffic sign name from the built-in dictionary
                traffic_sign_name = traffic_sign_names[predicted_class]

                # Display the predicted class label, traffic sign name, and probability
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text = f"Class: {predicted_class} - Sign: {traffic_sign_name} - Prob: {probability:.4f}"
                cv2.putText(frame, text, (x, y - 10), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the frame with the detected traffic signs
        cv2.imshow("Traffic Sign Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
