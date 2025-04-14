# Face Recognition Application using OpenCV and Streamlit

This application demonstrates face recognition capabilities using OpenCV and Streamlit. It offers multiple functionalities including face recognition, age and gender detection, and the ability to add new faces to the recognition database.

## Features

- **Age & Gender Detection**: Detect faces in real-time and predict age and gender using pre-trained deep learning models
- **Face Recognition**: Recognize known faces using a K-Nearest Neighbors (KNN) algorithm
- **Add New Face**: Easily add new faces to the recognition database through the webcam

## Requirements

- Python 3.7+
- Webcam
- The required models (see below)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Face-Recognition-using-OpenCV.git
cd Face-Recognition-using-OpenCV
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files (if not already in the repository):
   - haarcascade_frontalface_alt.xml or haarcascade_frontalface_default.xml
   - opencv_face_detector.pbtxt
   - opencv_face_detector_uint8.pb
   - deploy_age.prototxt
   - age_net.caffemodel
   - deploy_gender.prototxt
   - gender_net.caffemodel

   Place all these files in the `Models` directory.

   > **Note**: Some model files might be large and not included in the repository. You can download them from:
   > - Age and Gender models: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
   > - Face detection models: https://github.com/opencv/opencv/tree/master/data/haarcascades

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Use the sidebar to navigate between different functionalities:
   - **About**: Information about the application
   - **Age & Gender Detection**: Real-time age and gender prediction
   - **Face Recognition**: Recognize known faces
   - **Add New Face**: Add a new face to the recognition database

### Adding a New Face

1. Select "Add New Face" from the sidebar
2. Enter a name for the face
3. Click "Start Collecting Face Data"
4. Position your face in front of the camera
5. The application will collect multiple samples of your face
6. Once complete, your face will be added to the recognition database

### Face Recognition

1. Select "Face Recognition" from the sidebar
2. The application will display recognized faces with their names
3. If no faces are recognized, you'll need to add faces first

## How it Works

### Age & Gender Detection

The application uses deep neural networks to detect faces and predict age and gender. The prediction models were trained on large datasets and can predict with reasonable accuracy.

### Face Recognition

The app uses the K-Nearest Neighbors (KNN) algorithm for face recognition. It compares face encodings to find the closest match in the database.

## Limitations

- The accuracy of age and gender prediction may vary based on lighting conditions, facial expressions, and other factors
- Face recognition works best with frontal faces in good lighting
- The models may have biases inherent from their training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for the computer vision libraries and pre-trained models
- Streamlit for the web application framework 