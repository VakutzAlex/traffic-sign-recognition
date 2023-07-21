import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
NUM_CLASSES = 43
IMG_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 10

def load_data(data_dir):
    images = []
    labels = []
    for folder in os.listdir(data_dir):
        label = int(folder)
        folder_path = os.path.join(data_dir, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    labels = to_categorical(labels, NUM_CLASSES)
    return images, labels

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_dir = "path/to/your/train/folder"
    images, labels = load_data(data_dir)
    images, labels = preprocess_data(images, labels)

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model()
    model.summary()

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation accuracy: {test_accuracy:.4f}')

    # Save the trained model
    model.save_weights("Traffic.h5")

if __name__ == "__main__":
    main()
